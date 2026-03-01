"""
scripts/05_evaluate.py
Step 5: Final consolidated evaluation across all models.

Loads all saved models, evaluates on test + generalization_test,
produces publication-ready plots and a summary table for the thesis.

Outputs (in outputs/plots/ and outputs/results/):
  - roc_curves.png           — ROC curves for all models on test set
  - confusion_matrices.png   — Confusion matrix grid
  - degradation_curves.png   — EER vs SNR and compression (robustness)
  - feature_importance.png   — SHAP / RF feature importance
  - final_summary.csv        — Master results table
  - final_summary.txt        — Formatted table ready to paste into thesis

Run: python scripts/05_evaluate.py
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe on Windows/Colab)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.cnn import SpeechCNN, SpeechCNNLSTM
from src.evaluation.metrics import compute_all_metrics, compute_eer
from src.utils.audio_utils import load_audio, add_white_noise, simulate_g711

from sklearn.metrics import roc_curve, confusion_matrix

PROC_DIR    = ROOT / cfg["paths"]["data_processed"]
MODELS_DIR  = ROOT / cfg["paths"]["outputs_models"]
RESULTS_DIR = ROOT / cfg["paths"]["outputs_results"]
PLOTS_DIR   = ROOT / cfg["paths"]["outputs_plots"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = cfg["project"]["seed"]
np.random.seed(SEED)

# ── Colour palette (consistent across all plots) ───────────────────────────────
COLOURS = {
    "SVM":               "#2196F3",
    "Random Forest":     "#4CAF50",
    "Gradient Boosting": "#FF9800",
    "CNN":               "#9C27B0",
    "CNN-LSTM":          "#F44336",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_classical(name):
    path = MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib"
    if not path.exists():
        return None, None
    artifact = joblib.load(path)
    return artifact["model"], artifact["threshold"]


def load_deep(model_name):
    path = MODELS_DIR / f"{model_name}_best.pt"
    if not path.exists():
        return None
    model = SpeechCNN() if model_name == "cnn" else SpeechCNNLSTM()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict_classical(model, threshold, X):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= threshold).astype(int)
    return pred, proba


def predict_deep(model, X_spec):
    """X_spec: numpy (N,1,128,128)"""
    device = torch.device("cpu")
    model  = model.to(device)
    ds     = torch.tensor(X_spec, dtype=torch.float32)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    all_probs, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds  = logits.argmax(1).numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
    return np.array(all_preds), np.array(all_probs)


# ── 1. ROC Curves ──────────────────────────────────────────────────────────────

def plot_roc_curves(results_by_model, y_test, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, split in enumerate(["test", "generalization_test"]):
        ax = axes[ax_idx]
        ax.plot([0,1], [0,1], 'k--', alpha=0.4, label='Random (AUC=0.50)')

        for name, data in results_by_model.items():
            if split not in data:
                continue
            y_true  = data[split]["y_true"]
            y_score = data[split]["y_score"]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc  = data[split]["metrics"]["roc_auc"]
            eer  = data[split]["metrics"]["eer"]
            col  = COLOURS.get(name, "gray")
            ax.plot(fpr, tpr, color=col, lw=2,
                    label=f"{name}  (AUC={auc:.3f}, EER={eer:.3f})")

        title = "Standard Test Set" if split == "test" else "Generalization Test\n(Edge-TTS, unseen system)"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_xlim([0,1]); ax.set_ylim([0,1])

    plt.suptitle("ROC Curves — AI Speech Detection", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── 2. Confusion Matrices ──────────────────────────────────────────────────────

def plot_confusion_matrices(results_by_model, save_path):
    models = list(results_by_model.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, models):
        if "test" not in results_by_model[name]:
            continue
        y_true = results_by_model[name]["test"]["y_true"]
        y_pred = results_by_model[name]["test"]["y_pred"]
        cm     = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Genuine', 'Synthetic'])
        ax.set_yticklabels(['Genuine', 'Synthetic'])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(name, fontweight='bold')

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                        ha='center', va='center', fontsize=9,
                        color='white' if cm_pct[i,j] > 60 else 'black')

    plt.suptitle("Confusion Matrices — Test Set", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── 3. Robustness: EER vs SNR ──────────────────────────────────────────────────

def robustness_evaluation(results_by_model, save_path):
    """
    Re-evaluate classical models on progressively noisier/compressed audio.
    Uses a random subset of test manifest for speed.
    """
    manifest_path = PROC_DIR / "test_manifest.json"
    if not manifest_path.exists():
        print("  [Robustness] test_manifest.json not found — skipping.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Sample up to 300 clips for speed
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(manifest), size=min(300, len(manifest)), replace=False)
    manifest_sample = [manifest[i] for i in sample]

    SNR_LEVELS  = [5, 10, 15, 20, 999]   # 999 = clean
    LABEL_MAP   = {5: "5dB", 10: "10dB", 15: "15dB", 20: "20dB", 999: "Clean"}

    from src.features.extractor import extract_all_handcrafted
    TARGET_SR = cfg["audio"]["target_sr"]
    N_MFCC    = cfg["features"]["n_mfcc"]

    print("  [Robustness] Extracting features at each SNR level...")
    snr_features = {}
    y_true = np.array([item["label"] for item in manifest_sample])

    for snr in SNR_LEVELS:
        feats = []
        for item in manifest_sample:
            try:
                audio, sr = load_audio(item["path"], target_sr=TARGET_SR)
                if snr != 999:
                    audio = add_white_noise(audio, snr_db=snr)
                hc = extract_all_handcrafted(audio, sr, n_mfcc=N_MFCC)
                feats.append(hc)
            except Exception:
                feats.append(np.zeros(257))
        snr_features[snr] = np.array(feats, dtype=np.float32)

    # Also test codec compression
    print("  [Robustness] Testing G.711 codec compression...")
    codec_feats = []
    for item in manifest_sample:
        try:
            audio, sr = load_audio(item["path"], target_sr=TARGET_SR)
            audio = simulate_g711(audio, sr=TARGET_SR)
            hc = extract_all_handcrafted(audio, sr, n_mfcc=N_MFCC)
            codec_feats.append(hc)
        except Exception:
            codec_feats.append(np.zeros(257))
    codec_features = np.array(codec_feats, dtype=np.float32)

    # Evaluate each classical model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name in ["SVM", "Random Forest", "Gradient Boosting"]:
        model, threshold = load_classical(name)
        if model is None:
            continue
        col = COLOURS.get(name, "gray")

        # SNR degradation
        snr_eers = []
        for snr in SNR_LEVELS:
            proba = model.predict_proba(snr_features[snr])[:, 1]
            pred  = (proba >= threshold).astype(int)
            m = compute_all_metrics(y_true, pred, proba)
            snr_eers.append(m["eer"])

        x_labels = [LABEL_MAP[s] for s in SNR_LEVELS]
        axes[0].plot(x_labels, snr_eers, marker='o', color=col, label=name, lw=2)

        # Codec result (single point)
        proba = model.predict_proba(codec_features)[:, 1]
        pred  = (proba >= threshold).astype(int)
        m_codec = compute_all_metrics(y_true, pred, proba)
        axes[1].bar(name, m_codec["eer"], color=col, alpha=0.8)

    axes[0].set_title("EER vs. Noise Level (White Gaussian Noise)", fontweight='bold')
    axes[0].set_xlabel("SNR Level"); axes[0].set_ylabel("EER (lower is better)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].invert_xaxis()   # left=noisy, right=clean

    axes[1].set_title("EER under G.711 Codec Compression", fontweight='bold')
    axes[1].set_ylabel("EER (lower is better)")
    axes[1].grid(alpha=0.3, axis='y')

    plt.suptitle("Robustness Evaluation — Banking Telephony Conditions",
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── 4. Feature Importance (Random Forest SHAP-lite) ───────────────────────────

def plot_feature_importance(save_path):
    model_artifact = MODELS_DIR / "random_forest.joblib"
    names_path     = PROC_DIR / "feature_names.json"

    if not model_artifact.exists() or not names_path.exists():
        print("  [Features] RF model or feature names not found — skipping.")
        return

    artifact = joblib.load(model_artifact)
    rf_model = artifact["model"]

    # Get the actual RF classifier from the pipeline
    clf = rf_model.named_steps.get("clf") or rf_model[-1]

    with open(names_path) as f:
        feature_names = json.load(f)

    importances = clf.feature_importances_
    top_n = 20
    top_idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    colours_bar = ["#2196F3" if "mfcc" in feature_names[i]
                   else "#FF9800" if any(k in feature_names[i]
                        for k in ["centroid","bandwidth","rolloff","zcr","hnr"])
                   else "#4CAF50"
                   for i in top_idx]

    bars = ax.barh(range(top_n), importances[top_idx][::-1],
                   color=colours_bar[::-1], alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx[::-1]], fontsize=9)
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.set_title("Top 20 Feature Importances — Random Forest\n"
                 "(Blue=MFCC, Orange=Spectral, Green=Pitch/Phase)",
                 fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── 5. Summary table ───────────────────────────────────────────────────────────

def build_summary_table(results_by_model):
    rows = []
    for model_name, splits in results_by_model.items():
        for split_name, data in splits.items():
            m = data["metrics"]
            rows.append({
                "Model":    model_name,
                "Split":    split_name,
                "EER":      round(m["eer"],      4),
                "F1":       round(m["f1"],       4),
                "ROC-AUC":  round(m["roc_auc"],  4),
                "Accuracy": round(m["accuracy"], 4),
                "Precision":round(m["precision"],4),
                "Recall":   round(m["recall"],   4),
            })

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "final_summary.csv"
    df.to_csv(csv_path, index=False)

    # Pretty-print version for thesis
    txt_path = RESULTS_DIR / "final_summary.txt"
    with open(txt_path, "w") as f:
        f.write("AI-Generated Speech Detection — Results Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\nNote: EER = Equal Error Rate (primary metric, lower is better)\n")
        f.write("      Test = standard held-out set (all TTS sources)\n")
        f.write("      generalization_test = Edge-TTS only (unseen TTS system)\n")

    print(f"  Saved: final_summary.csv  +  final_summary.txt")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 5: Final Evaluation & Plots")
    print("=" * 60)

    # Load test data
    X_hc_test   = np.load(PROC_DIR / "X_handcrafted_test.npy")
    X_spec_test = np.load(PROC_DIR / "X_spectrogram_test.npy")
    y_test      = np.load(PROC_DIR / "y_test.npy")

    X_hc_gen, X_spec_gen, y_gen = None, None, None
    if (PROC_DIR / "X_handcrafted_generalization_test.npy").exists():
        X_hc_gen   = np.load(PROC_DIR / "X_handcrafted_generalization_test.npy")
        X_spec_gen = np.load(PROC_DIR / "X_spectrogram_generalization_test.npy")
        y_gen      = np.load(PROC_DIR / "y_generalization_test.npy")

    results_by_model = defaultdict(dict)

    # ── Classical models ───────────────────────────────────────────────────────
    for name in ["SVM", "Random Forest", "Gradient Boosting"]:
        model, threshold = load_classical(name)
        if model is None:
            print(f"  [{name}] Not found — skipping.")
            continue

        for split, X_hc, y in [
            ("test",               X_hc_test, y_test),
            ("generalization_test", X_hc_gen, y_gen),
        ]:
            if X_hc is None:
                continue
            pred, proba = predict_classical(model, threshold, X_hc)
            m = compute_all_metrics(y, pred, proba)
            results_by_model[name][split] = {
                "y_true": y, "y_pred": pred, "y_score": proba, "metrics": m
            }
            print(f"  [{name:20s}] [{split:20s}] EER={m['eer']:.4f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")

    # ── Deep models ───────────────────────────────────────────────────────────
    for model_name, display_name in [("cnn", "CNN"), ("cnn_lstm", "CNN-LSTM")]:
        model = load_deep(model_name)
        if model is None:
            print(f"  [{display_name}] Checkpoint not found — skipping.")
            continue

        for split, X_spec, y in [
            ("test",               X_spec_test, y_test),
            ("generalization_test", X_spec_gen, y_gen),
        ]:
            if X_spec is None:
                continue
            pred, proba = predict_deep(model, X_spec)
            m = compute_all_metrics(y, pred, proba)
            results_by_model[display_name][split] = {
                "y_true": y, "y_pred": pred, "y_score": proba, "metrics": m
            }
            print(f"  [{display_name:20s}] [{split:20s}] EER={m['eer']:.4f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")

    if not results_by_model:
        print("\nNo models found. Train models first, then run this script.")
        sys.exit(1)

    # ── Generate all plots ────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_roc_curves(results_by_model, y_test,
                    PLOTS_DIR / "roc_curves.png")
    plot_confusion_matrices(results_by_model,
                             PLOTS_DIR / "confusion_matrices.png")
    plot_feature_importance(PLOTS_DIR / "feature_importance.png")

    print("\nRunning robustness evaluation (takes ~5 min)...")
    robustness_evaluation(results_by_model,
                           PLOTS_DIR / "degradation_curves.png")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\nBuilding summary table...")
    df = build_summary_table(results_by_model)
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n✓ All outputs saved to {PLOTS_DIR} and {RESULTS_DIR}")
    print("  Thesis-ready table: outputs/results/final_summary.txt")
