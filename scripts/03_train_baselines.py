"""
scripts/03_train_baselines.py  [v2 — imbalance-aware]
Step 3: Train SVM, Random Forest, and Gradient Boosting classifiers.

Fixes applied vs v1:
  1. class_weight='balanced' on all models — forces equal attention to minority class
  2. SMOTE oversampling on training set — synthetically augments synthetic speech samples
  3. Threshold tuning on val set — moves decision boundary from 0.5 to EER-optimal point
  4. Scoring changed from 'f1' to 'roc_auc' in grid search — AUC is imbalance-robust

Run: python scripts/03_train_baselines.py
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve

from src.evaluation.metrics import compute_all_metrics

PROC_DIR    = ROOT / cfg["paths"]["data_processed"]
MODELS_DIR  = ROOT / cfg["paths"]["outputs_models"]
RESULTS_DIR = ROOT / cfg["paths"]["outputs_results"]
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = cfg["project"]["seed"]
np.random.seed(SEED)


# Install imbalanced-learn if needed
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    import subprocess
    print("Installing imbalanced-learn...")
    subprocess.run([sys.executable, "-m", "pip", "install", "imbalanced-learn", "-q"])
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline


def load_split(split):
    X = np.load(PROC_DIR / f"X_handcrafted_{split}.npy")
    y = np.load(PROC_DIR / f"y_{split}.npy")
    return X, y


def find_optimal_threshold(y_true, y_score):
    """
    Find probability threshold at the EER point (where FPR == FNR).
    Using 0.5 is wrong under heavy imbalance — the model's score range shifts.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float(thresholds[idx])


def evaluate_with_threshold(model, X, y, threshold):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= threshold).astype(int)
    return compute_all_metrics(y, pred, proba)


def train_and_evaluate(name, pipeline, param_grid,
                        X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n[{name}] Running grid search (scoring=roc_auc)...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipeline, param_grid, cv=cv,
                      scoring="roc_auc", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")

    # Find optimal threshold on val set
    val_proba = best.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold(y_val, val_proba)
    print(f"  Optimal threshold (EER point on val): {threshold:.4f}  [default was 0.5]")

    results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        metrics = evaluate_with_threshold(best, X, y, threshold)
        metrics["split"]     = split_name
        metrics["model"]     = name
        metrics["threshold"] = threshold
        results[split_name]  = metrics

        proba = best.predict_proba(X)[:, 1]
        pred  = (proba >= threshold).astype(int)
        pc = Counter(pred.tolist())
        tc = Counter(y.tolist())
        print(f"  [{split_name}] EER={metrics['eer']:.4f} | F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f} | Acc={metrics['accuracy']:.4f}")
        print(f"           Predicted: genuine={pc.get(0,0)} synthetic={pc.get(1,0)}"
              f" | Actual: genuine={tc.get(0,0)} synthetic={tc.get(1,0)}")

    artifact = {"model": best, "threshold": threshold}
    model_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib"
    joblib.dump(artifact, model_path)
    print(f"  Saved: {model_path}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Step 3: Baseline ML Training  [imbalance-aware v2]")
    print("=" * 60)

    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    c = Counter(y_train.tolist())
    k = min(5, c[1] - 1)  # SMOTE k_neighbors must be < n_minority_samples
    print(f"\nTrain: genuine={c[0]} synthetic={c[1]} (ratio {c[0]//max(c[1],1)}:1)")
    print(f"Strategy: SMOTE (k={k}) + class_weight=balanced + EER threshold tuning\n")

    # SVM
    svm_pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=SEED, k_neighbors=k)),
        ("clf",    SVC(probability=True, class_weight="balanced", random_state=SEED))
    ])
    svm_params = {
        "clf__C":      cfg["models"]["classical"]["svm"]["C"],
        "clf__gamma":  cfg["models"]["classical"]["svm"]["gamma"],
        "clf__kernel": [cfg["models"]["classical"]["svm"]["kernel"]],
    }

    # Random Forest
    rf_pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=SEED, k_neighbors=k)),
        ("clf",    RandomForestClassifier(class_weight="balanced",
                                          random_state=SEED, n_jobs=-1))
    ])
    rf_params = {
        "clf__n_estimators": cfg["models"]["classical"]["random_forest"]["n_estimators"],
        "clf__max_depth":    cfg["models"]["classical"]["random_forest"]["max_depth"],
    }

    # Gradient Boosting (no native class_weight — SMOTE handles balance)
    gb_pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=SEED, k_neighbors=k)),
        ("clf",    GradientBoostingClassifier(random_state=SEED))
    ])
    gb_params = {
        "clf__n_estimators":  cfg["models"]["classical"]["gradient_boosting"]["n_estimators"],
        "clf__learning_rate": cfg["models"]["classical"]["gradient_boosting"]["learning_rate"],
    }

    all_results = []
    for name, pipe, params in [
        ("SVM",               svm_pipeline, svm_params),
        ("Random Forest",     rf_pipeline,  rf_params),
        ("Gradient Boosting", gb_pipeline,  gb_params),
    ]:
        res = train_and_evaluate(name, pipe, params,
                                  X_train, y_train,
                                  X_val, y_val,
                                  X_test, y_test)
        all_results.extend(res.values())

    df = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / "baseline_results.csv"
    df.to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(df[["model", "split", "eer", "f1", "roc_auc", "accuracy"]].to_string(index=False))
    print(f"\n✓ Results saved to {results_path}")
    print("✓ Step 3 complete. Run scripts/04_train_cnn.py next.")