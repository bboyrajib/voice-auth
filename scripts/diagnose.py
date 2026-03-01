"""
scripts/diagnose.py
Run this to understand exactly what's happening with your data and models.
Prints class distribution, checks feature quality, and shows what models are predicting.
"""

import sys
import json
import numpy as np
import joblib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

PROC_DIR   = ROOT / cfg["paths"]["data_processed"]
MODELS_DIR = ROOT / cfg["paths"]["outputs_models"]

print("=" * 60)
print("DIAGNOSIS REPORT")
print("=" * 60)

# ── 1. Class distribution ──────────────────────────────────────────────────────
print("\n[1] CLASS DISTRIBUTION")
for split in ["train", "val", "test"]:
    y = np.load(PROC_DIR / f"y_{split}.npy")
    counts = Counter(y.tolist())
    total = len(y)
    n0, n1 = counts.get(0, 0), counts.get(1, 0)
    ratio = n0 / n1 if n1 > 0 else float('inf')
    print(f"  {split:6s}: {total:4d} total | genuine={n0} ({100*n0/total:.1f}%) | synthetic={n1} ({100*n1/total:.1f}%) | ratio={ratio:.1f}:1")

# ── 2. What the best model actually predicts ───────────────────────────────────
print("\n[2] MODEL PREDICTION DISTRIBUTION")
for model_name in ["svm", "random_forest", "gradient_boosting"]:
    mpath = MODELS_DIR / f"{model_name}.joblib"
    if not mpath.exists():
        continue
    model = joblib.load(mpath)
    for split in ["val", "test"]:
        X = np.load(PROC_DIR / f"X_handcrafted_{split}.npy")
        y = np.load(PROC_DIR / f"y_{split}.npy")
        preds = model.predict(X)
        pred_counts = Counter(preds.tolist())
        print(f"  {model_name:20s} [{split}]: predicts genuine={pred_counts.get(0,0)} synthetic={pred_counts.get(1,0)} | actual genuine={Counter(y.tolist())[0]} synthetic={Counter(y.tolist()).get(1,0)}")

# ── 3. Feature sanity check ────────────────────────────────────────────────────
print("\n[3] FEATURE SANITY CHECK")
X_train = np.load(PROC_DIR / "X_handcrafted_train.npy")
y_train = np.load(PROC_DIR / "y_train.npy")

X_genuine   = X_train[y_train == 0]
X_synthetic = X_train[y_train == 1]

print(f"  Feature vector length: {X_train.shape[1]}")
print(f"  NaN count: {np.isnan(X_train).sum()}")
print(f"  Inf count: {np.isinf(X_train).sum()}")

if len(X_synthetic) > 0:
    # Find most discriminative features (largest mean difference)
    diff = np.abs(X_genuine.mean(axis=0) - X_synthetic.mean(axis=0))
    top5_idx = diff.argsort()[::-1][:5]

    try:
        with open(PROC_DIR / "feature_names.json") as f:
            names = json.load(f)
        print(f"\n  Top 5 most different features between genuine and synthetic:")
        for i in top5_idx:
            print(f"    [{i:3d}] {names[i]:35s} genuine_mean={X_genuine[:,i].mean():.3f} | synthetic_mean={X_synthetic[:,i].mean():.3f} | diff={diff[i]:.3f}")
    except FileNotFoundError:
        print(f"  Top 5 most different feature indices: {top5_idx}")
else:
    print("  WARNING: No synthetic samples found in training set!")

# ── 4. Root cause summary ──────────────────────────────────────────────────────
print("\n[4] ROOT CAUSE SUMMARY")
y_train = np.load(PROC_DIR / "y_train.npy")
n0, n1 = (y_train==0).sum(), (y_train==1).sum()

if n1 == 0:
    print("  CRITICAL: Zero synthetic samples in training set. Models never learned to detect synthetic speech.")
elif n0 / n1 > 10:
    print(f"  CRITICAL: Severe class imbalance — {n0} genuine vs {n1} synthetic ({n0/n1:.0f}:1 ratio).")
    print("  Models are predicting almost everything as genuine to maximize accuracy.")
    print("  Fix: Use class_weight='balanced' + generate more synthetic data.")
elif n0 / n1 > 3:
    print(f"  WARNING: Moderate class imbalance — {n0} genuine vs {n1} synthetic ({n0/n1:.1f}:1 ratio).")
    print("  Fix: Use class_weight='balanced' in models.")
else:
    print(f"  Class balance OK: {n0} genuine vs {n1} synthetic.")
    print("  The issue may be feature quality or model overfitting.")

print("\nDone. Share this output and we'll fix the exact issue.")
