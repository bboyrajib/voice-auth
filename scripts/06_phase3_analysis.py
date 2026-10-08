"""
scripts/06_phase3_analysis.py
Step 6 (Phase 3): Interpretability analysis, final evaluation and statistical testing.

Implements the Phase 3 plan from the Phase 2 report ("Revised Plan for Phase 3"):

  Part A  shap        — SHAP for the classical models (SVM, RF, Gradient Boosting),
                        cross-checked against the Phase 2 Random Forest importances.
  Part B  saliency    — Gradient saliency mapping for CNN / CNN-LSTM
                        (vanilla gradient, SmoothGrad, Integrated Gradients) with a
                        deletion (faithfulness) test against a random baseline.
  Part C  stats       — Final evaluation with statistical testing: bootstrap 95% CIs,
                        pairwise McNemar (Holm-corrected), DeLong AUC test and a
                        paired-bootstrap EER difference test; DET curves.
  Part D  robustness  — Noise / G.711 robustness for ALL five models (Phase 2 only
                        covered the classical models), feeding the noise-hardening
                        agenda.

Outputs (in outputs/phase3/plots/ and outputs/phase3/results/).

Run:
  python scripts/06_phase3_analysis.py                     # all parts
  python scripts/06_phase3_analysis.py --parts shap stats  # selected parts
  python scripts/06_phase3_analysis.py --quick             # small samples, smoke test
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe on Windows/Colab)
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from itertools import combinations

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

import torch
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score

from src.models.cnn import SpeechCNN, SpeechCNNLSTM
from src.evaluation.metrics import compute_all_metrics, compute_eer

PROC_DIR    = ROOT / cfg["paths"]["data_processed"]
MODELS_DIR  = ROOT / cfg["paths"]["outputs_models"]
OUT_DIR     = ROOT / "outputs" / "phase3"
PLOTS_DIR   = OUT_DIR / "plots"
RESULTS_DIR = OUT_DIR / "results"
CACHE_DIR   = OUT_DIR / "cache"
for d in (PLOTS_DIR, RESULTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED      = cfg["project"]["seed"]
TARGET_SR = cfg["audio"]["target_sr"]
N_MELS    = cfg["features"]["n_mels"]
N_FFT     = cfg["features"]["n_fft"]
HOP       = cfg["features"]["hop_length"]
N_MFCC    = cfg["features"]["n_mfcc"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Colour palette (same as 05_evaluate.py) ────────────────────────────────────
COLOURS = {
    "SVM":               "#2196F3",
    "Random Forest":     "#4CAF50",
    "Gradient Boosting": "#FF9800",
    "CNN":               "#9C27B0",
    "CNN-LSTM":          "#F44336",
}
CLASSICAL = ["SVM", "Random Forest", "Gradient Boosting"]
DEEP      = {"CNN": "cnn", "CNN-LSTM": "cnn_lstm"}
ALL_MODELS = CLASSICAL + list(DEEP)
SPLITS = {"test": "Standard test", "generalization_test": "Generalisation test (Edge-TTS)"}
CLASS_NAMES = ["Genuine", "Synthetic"]


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def savefig(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: plots/{name}")


def load_classical(name):
    artifact = joblib.load(MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")
    return artifact["model"], artifact["threshold"]


def load_deep(display_name):
    model = SpeechCNN() if DEEP[display_name] == "cnn" else SpeechCNNLSTM()
    state = torch.load(MODELS_DIR / f"{DEEP[display_name]}_best.pt", map_location="cpu")
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def load_split(split):
    return (np.load(PROC_DIR / f"X_handcrafted_{split}.npy"),
            np.load(PROC_DIR / f"X_spectrogram_{split}.npy"),
            np.load(PROC_DIR / f"y_{split}.npy"))


def predict_deep(model, X_spec, batch_size=64):
    """Returns (pred, P(synthetic)). pred = argmax, as in Phase 2."""
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_spec), batch_size):
            xb = torch.as_tensor(X_spec[i:i + batch_size], dtype=torch.float32, device=DEVICE)
            probs.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    proba = np.concatenate(probs)
    return (proba >= 0.5).astype(int), proba


def predict_all(models, X_hc, X_spec):
    """models: dict name -> (model, threshold|None). Returns name -> (pred, score)."""
    out = {}
    for name, (model, thr) in models.items():
        if name in CLASSICAL:
            proba = model.predict_proba(X_hc)[:, 1]
            out[name] = ((proba >= thr).astype(int), proba)
        else:
            out[name] = predict_deep(model, X_spec)
    return out


def load_all_models():
    models = {}
    for name in CLASSICAL:
        models[name] = load_classical(name)
    for name in DEEP:
        models[name] = (load_deep(name), None)
    return models


def balanced_sample(y, n, rng):
    """n indices, half from each class (or as many as available)."""
    idx = []
    for c in (0, 1):
        pool = np.flatnonzero(y == c)
        idx.append(rng.choice(pool, size=min(n // 2, len(pool)), replace=False))
    return np.sort(np.concatenate(idx))


def mel_bin_hz():
    import librosa
    return librosa.mel_frequencies(n_mels=N_MELS, fmin=0.0, fmax=TARGET_SR / 2)


def frame_seconds(n_frames):
    return np.arange(n_frames) * HOP / TARGET_SR


# ══════════════════════════════════════════════════════════════════════════════
# Part A — SHAP for the classical models
# ══════════════════════════════════════════════════════════════════════════════

SPECTRAL_KEYS = ("centroid", "bandwidth", "rolloff", "zcr", "hnr")
GROUP_ORDER = ["MFCC", "Δ-MFCC", "Δ²-MFCC", "Spectral", "Pitch", "Phase"]
GROUP_COLOURS = dict(zip(GROUP_ORDER,
                         ["#90CAF9", "#1E88E5", "#0D47A1", "#FF9800", "#4CAF50", "#8E24AA"]))


def feature_group(name):
    if name.startswith("delta2_mfcc"):
        return "Δ²-MFCC"
    if name.startswith("delta_mfcc"):
        return "Δ-MFCC"
    if name.startswith("mfcc"):
        return "MFCC"
    if name.startswith(SPECTRAL_KEYS):
        return "Spectral"
    if name.startswith(("f0", "voiced")):
        return "Pitch"
    return "Phase"


def _positive_class(values):
    """Normalise SHAP output to (n_samples, n_features) for the synthetic class."""
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    return values[..., 1] if values.ndim == 3 else values


def run_shap(args):
    import shap

    log("=" * 60)
    log("Part A — SHAP interpretability (classical models)")
    log("=" * 60)
    with open(PROC_DIR / "feature_names.json") as f:
        names = json.load(f)
    groups = np.array([feature_group(n) for n in names])

    X_tr, _, y_tr = load_split("train")
    X_te, _, y_te = load_split("test")
    rng = np.random.default_rng(SEED)
    idx_tree = balanced_sample(y_te, args.shap_n, rng)
    idx_svm  = balanced_sample(y_te, args.shap_svm_n, rng)
    idx_bg   = balanced_sample(y_tr, args.shap_background, rng)

    shap_vals, shap_X = {}, {}
    for name in CLASSICAL:
        pipe, _ = load_classical(name)
        scaler, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
        t0 = time.time()
        if name == "SVM":
            # Model-agnostic permutation SHAP on P(synthetic) in standardised space.
            Xs = scaler.transform(X_te[idx_svm])
            bg = scaler.transform(X_tr[idx_bg])
            f = lambda z: clf.predict_proba(z)[:, 1]
            explainer = shap.PermutationExplainer(f, shap.maskers.Independent(bg))
            vals = explainer(Xs, max_evals=2 * len(names) + 1, silent=True).values
            unit = "Δ P(synthetic)"
        else:
            # Exact TreeSHAP. RF -> probability units, GB -> log-odds units.
            Xs = scaler.transform(X_te[idx_tree])
            explainer = shap.TreeExplainer(clf)
            batches = []
            for i in range(0, len(Xs), 25):
                batches.append(_positive_class(
                    explainer.shap_values(Xs[i:i + 25], check_additivity=False)))
                log(f"  [{name}] {min(i + 25, len(Xs))}/{len(Xs)} samples "
                    f"({time.time() - t0:.0f}s)")
            vals = np.concatenate(batches)
            unit = "Δ P(synthetic)" if name == "Random Forest" else "Δ log-odds"
        shap_vals[name], shap_X[name] = vals, Xs
        log(f"  [{name}] SHAP {vals.shape} in {time.time() - t0:.0f}s ({unit})")

        # Beeswarm (top 20)
        fig = plt.figure()
        shap.summary_plot(vals, Xs, feature_names=names, max_display=20, show=False,
                          plot_size=(9, 8))
        plt.title(f"SHAP summary — {name}\n(x: {unit}; colour: standardised feature value)",
                  fontweight="bold")
        savefig(plt.gcf(), f"shap_beeswarm_{name.lower().replace(' ', '_')}.png")

    np.savez_compressed(RESULTS_DIR / "shap_values.npz",
                        feature_names=np.array(names),
                        **{k.replace(" ", "_"): v for k, v in shap_vals.items()})

    # ── Global importance table: mean |SHAP| vs RF impurity importance ────────
    rf_pipe, _ = load_classical("Random Forest")
    mdi = rf_pipe.named_steps["clf"].feature_importances_
    imp = pd.DataFrame({"feature": names, "group": groups, "RF MDI (Phase 2)": mdi})
    for name in CLASSICAL:
        imp[f"{name} |SHAP|"] = np.abs(shap_vals[name]).mean(axis=0)
    cols = [f"{n} |SHAP|" for n in CLASSICAL] + ["RF MDI (Phase 2)"]
    for c in cols:
        imp[f"{c} rank"] = imp[c].rank(ascending=False).astype(int)
    imp.sort_values("SVM |SHAP|", ascending=False).to_csv(
        RESULTS_DIR / "shap_global_importance.csv", index=False)

    # ── Cross-check: rank agreement between methods ───────────────────────────
    labels = [c.replace(" |SHAP|", " SHAP") for c in cols]
    k = 20
    rho = np.zeros((len(cols), len(cols)))
    overlap = np.zeros_like(rho)
    rows = []
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            rho[i, j] = stats.spearmanr(imp[a], imp[b]).statistic
            top_a = set(imp.nlargest(k, a)["feature"])
            top_b = set(imp.nlargest(k, b)["feature"])
            overlap[i, j] = len(top_a & top_b) / k
            if i < j:
                rows.append({"A": labels[i], "B": labels[j],
                             "spearman_rho": round(rho[i, j], 4),
                             f"top{k}_overlap": round(overlap[i, j], 3)})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "shap_rank_agreement.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, title, fmt in [(axes[0], rho, "Spearman ρ (all 257 features)", "{:.2f}"),
                                (axes[1], overlap, f"Top-{k} feature overlap", "{:.0%}")]:
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > 0.6 else "black", fontsize=10)
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Agreement between SHAP rankings and Phase 2 RF importances",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "shap_rank_agreement.png")

    # ── Top-20 comparison (normalised so each method sums to 1) ───────────────
    norm = imp[cols].div(imp[cols].sum(axis=0), axis=1)
    top = norm.mean(axis=1).nlargest(20).index[::-1]
    fig, ax = plt.subplots(figsize=(10, 9))
    h = 0.2
    for m, c in enumerate(cols):
        colour = COLOURS.get(c.replace(" |SHAP|", ""), "#616161")
        ax.barh(np.arange(len(top)) + (m - 1.5) * h, norm.loc[top, c], height=h,
                color=colour, label=labels[m],
                hatch="//" if c.startswith("RF MDI") else None, alpha=0.9)
    ax.set_yticks(range(len(top)), imp.loc[top, "feature"], fontsize=9)
    ax.set_xlabel("Normalised importance (share of total per method)")
    ax.set_title("Top 20 features — SHAP (3 models) vs Phase 2 RF importance",
                 fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    savefig(fig, "shap_top20_comparison.png")

    # ── Feature-group shares ──────────────────────────────────────────────────
    group_share = norm.groupby(imp["group"]).sum().reindex(GROUP_ORDER)
    group_share.columns = labels
    group_share.to_csv(RESULTS_DIR / "shap_group_importance.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(GROUP_ORDER))
    for m, lab in enumerate(labels):
        colour = COLOURS.get(lab.replace(" SHAP", ""), "#616161")
        ax.bar(x + (m - 1.5) * 0.2, group_share[lab], width=0.2, color=colour, label=lab,
               hatch="//" if lab.startswith("RF MDI") else None)
    n_feats = pd.Series(groups).value_counts().reindex(GROUP_ORDER)
    ax.set_xticks(x, [f"{g}\n({n} feats)" for g, n in n_feats.items()])
    ax.set_ylabel("Share of total importance")
    ax.set_title("Importance by feature group", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    savefig(fig, "shap_group_importance.png")

    # ── Dependence plots: top-3 SVM features ──────────────────────────────────
    top3 = imp.nlargest(3, "SVM |SHAP|").index
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, fi in zip(axes, top3):
        for name in CLASSICAL:
            ax.scatter(shap_X[name][:, fi], shap_vals[name][:, fi] /
                       (np.abs(shap_vals[name]).max() + 1e-12),
                       s=8, alpha=0.5, color=COLOURS[name], label=name)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(f"{names[fi]} (standardised)")
        ax.set_ylabel("SHAP (scaled to model max)")
        ax.grid(alpha=0.3)
    axes[0].legend(markerscale=3)
    fig.suptitle("SHAP dependence — top 3 SVM features (positive → pushes towards synthetic)",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "shap_dependence_top3.png")

    log("  Top-10 features by mean |SHAP|:")
    for name in CLASSICAL:
        top10 = imp.nlargest(10, f"{name} |SHAP|")["feature"].tolist()
        log(f"    {name:18s}: {', '.join(top10)}")
    log("  Rank agreement with RF MDI:")
    for r in rows:
        if r["B"] == "RF MDI (Phase 2)":
            log(f"    {r['A']:24s} ρ={r['spearman_rho']:.3f}  "
                f"top-{k} overlap={r[f'top{k}_overlap']:.0%}")
    log("  Feature-group shares:\n" + group_share.round(3).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# Part B — Gradient saliency for CNN / CNN-LSTM
# ══════════════════════════════════════════════════════════════════════════════

def _log_odds(model, x):
    """Synthetic-vs-genuine logit difference — the quantity being explained."""
    logits = model(x)
    return logits[:, 1] - logits[:, 0]


def _grad(model, x):
    x = x.detach().requires_grad_(True)
    return torch.autograd.grad(_log_odds(model, x).sum(), x)[0]


def saliency_maps(model, X, args, batch_size=16):
    """Returns dict method -> (N,128,128) attribution, plus IG completeness error."""
    out = {"Vanilla gradient": [], "SmoothGrad": [], "Integrated Gradients": []}
    completeness = []
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    alphas = torch.linspace(1.0 / args.ig_steps, 1.0, args.ig_steps, device=DEVICE)
    # cuDNN's LSTM backward only runs in train mode; disable it so eval-mode
    # gradients work on GPU too (no-op on CPU).
    with torch.backends.cudnn.flags(enabled=False):
        for i in range(0, len(X), batch_size):
            x = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=DEVICE)
            out["Vanilla gradient"].append(_grad(model, x).abs())

            span = (x.amax(dim=(1, 2, 3), keepdim=True) - x.amin(dim=(1, 2, 3), keepdim=True))
            sg = torch.zeros_like(x)
            for _ in range(args.smoothgrad_n):
                noise = torch.randn(x.shape, generator=gen, device=DEVICE) * 0.15 * span
                sg += _grad(model, x + noise).abs()
            out["SmoothGrad"].append(sg / args.smoothgrad_n)

            # Integrated Gradients from an all-zero baseline (= the per-clip mean
            # of the normalised log-Mel spectrogram, i.e. a flat spectrum).
            baseline = torch.zeros_like(x)
            total = torch.zeros_like(x)
            for a in alphas:
                total += _grad(model, baseline + a * (x - baseline))
            ig = (x - baseline) * total / args.ig_steps
            out["Integrated Gradients"].append(ig)
            with torch.no_grad():
                delta = _log_odds(model, x) - _log_odds(model, baseline)
                completeness.append(((ig.sum(dim=(1, 2, 3)) - delta).abs() /
                                     (delta.abs() + 1e-6)).cpu().numpy())
    maps = {k: torch.cat(v)[:, 0].detach().cpu().numpy() for k, v in out.items()}
    return maps, float(np.median(np.concatenate(completeness)))


def deletion_test(model, X, y, maps, fractions, rng, batch_size=64):
    """
    Faithfulness check: replace the top-k% most salient cells with the baseline
    value (0) and track the mean true-class probability. A faithful saliency map
    should make it fall faster than deleting the same number of random cells.
    """
    rankings = {m: np.argsort(-np.abs(a).reshape(len(a), -1), axis=1) for m, a in maps.items()}
    n_cells = X[0].size
    rankings["Random"] = np.stack([rng.permutation(n_cells) for _ in range(len(X))])
    rows = []
    for method, order in rankings.items():
        for frac in fractions:
            k = int(round(frac * n_cells))
            Xd = X.reshape(len(X), -1).copy()
            if k:
                np.put_along_axis(Xd, order[:, :k], 0.0, axis=1)
            _, p_syn = predict_deep(model, Xd.reshape(X.shape), batch_size)
            p_true = np.where(y == 1, p_syn, 1 - p_syn)
            rows.append({"method": method, "fraction_deleted": frac,
                         "mean_true_class_prob": float(p_true.mean()),
                         "accuracy": float(((p_syn >= 0.5).astype(int) == y).mean())})
    return pd.DataFrame(rows)


def run_saliency(args):
    log("=" * 60)
    log(f"Part B — Gradient saliency (CNN / CNN-LSTM) on {DEVICE}")
    log("=" * 60)
    _, X_spec, y = load_split("test")
    hz = mel_bin_hz()
    secs = frame_seconds(X_spec.shape[-1])
    rng = np.random.default_rng(SEED)
    fractions = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]

    methods = ["Vanilla gradient", "SmoothGrad", "Integrated Gradients"]
    class_mean, freq_rows, deletion_frames, summary = {}, [], [], []
    for name in DEEP:
        model = load_deep(name)
        pred, proba = predict_deep(model, X_spec)
        # Explain correctly-classified clips only, balanced across classes.
        correct = pred == y
        idx = np.sort(np.concatenate([
            rng.choice(np.flatnonzero((y == c) & correct),
                       size=min(args.saliency_n, int(((y == c) & correct).sum())),
                       replace=False) for c in (0, 1)]))
        Xs, ys = X_spec[idx], y[idx]
        t0 = time.time()
        maps, ig_err = saliency_maps(model, Xs, args)
        log(f"  [{name}] {len(idx)} clips, 3 methods in {time.time() - t0:.0f}s "
            f"(IG completeness error, median: {ig_err:.2%})")
        np.savez_compressed(RESULTS_DIR / f"saliency_{DEEP[name]}.npz",
                            indices=idx, labels=ys,
                            **{m.replace(" ", "_"): a for m, a in maps.items()})

        # Normalise each map to sum 1 so every clip contributes equally.
        norm = {m: np.abs(a) / (np.abs(a).sum(axis=(1, 2), keepdims=True) + 1e-12)
                for m, a in maps.items()}
        class_mean[name] = {c: {m: norm[m][ys == c].mean(axis=0) for m in methods}
                            for c in (0, 1)}

        # Per-band share of attribution (SmoothGrad)
        bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4001)]
        for c in (0, 1):
            prof = class_mean[name][c]["SmoothGrad"].sum(axis=1)
            for lo, hi in bands:
                mask = (hz >= lo) & (hz < hi)
                freq_rows.append({"model": name, "class": CLASS_NAMES[c],
                                  "band_hz": f"{lo}-{min(hi, 4000)}",
                                  "n_mel_bins": int(mask.sum()),
                                  "attribution_share": round(float(prof[mask].sum()), 4)})

        # ── Example maps ──────────────────────────────────────────────────────
        examples = []
        for c in (0, 1):
            pool = np.flatnonzero(ys == c)
            conf = np.where(c == 1, proba[idx][pool], 1 - proba[idx][pool])
            examples += list(pool[np.argsort(-conf)[:2]])
        fig, axes = plt.subplots(len(examples), 4, figsize=(18, 3.6 * len(examples)))
        extent = [secs[0], secs[-1], 0, N_MELS]
        yt = np.linspace(0, N_MELS - 1, 5).astype(int)
        for r, e in enumerate(examples):
            panels = [("Log-Mel spectrogram", Xs[e, 0], "magma", None)]
            for m in methods:
                a = maps[m][e]
                if m == "Integrated Gradients":
                    v = np.percentile(np.abs(a), 99.5)
                    panels.append((f"{m} (red → synthetic)", a, "RdBu_r", (-v, v)))
                else:
                    panels.append((m, a, "inferno", (0, np.percentile(a, 99.5))))
            for ax, (title, img, cmap, lim) in zip(axes[r], panels):
                ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, extent=extent,
                          vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
                ax.set_yticks(yt, [f"{hz[t]:.0f}" for t in yt])
                if r == 0:
                    ax.set_title(title, fontweight="bold")
                if ax is axes[r][0]:
                    ax.set_ylabel(f"{CLASS_NAMES[ys[e]]} #{idx[e]}\nFrequency (Hz)")
                if r == len(examples) - 1:
                    ax.set_xlabel("Time (s)")
        fig.suptitle(f"Gradient saliency — {name} (correctly classified test clips)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        savefig(fig, f"saliency_examples_{DEEP[name]}.png")

        # ── Deletion test ────────────────────────────────────────────────────
        t0 = time.time()
        df = deletion_test(model, Xs, ys, maps, fractions, rng)
        df.insert(0, "model", name)
        deletion_frames.append(df)
        log(f"  [{name}] deletion test in {time.time() - t0:.0f}s")
        at10 = df[df.fraction_deleted == 0.10].set_index("method")["mean_true_class_prob"]
        summary.append({"model": name, "ig_completeness_err": ig_err,
                        **{f"P(true) after 10% deletion — {m}": round(v, 4)
                           for m, v in at10.items()}})

    # ── Class-mean maps ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for r, name in enumerate(DEEP):
        diff = class_mean[name][1]["SmoothGrad"] - class_mean[name][0]["SmoothGrad"]
        v = np.abs(diff).max()
        panels = [(f"{name} — genuine", class_mean[name][0]["SmoothGrad"], "inferno", None),
                  (f"{name} — synthetic", class_mean[name][1]["SmoothGrad"], "inferno", None),
                  (f"{name} — synthetic − genuine", diff, "RdBu_r", (-v, v))]
        for ax, (title, img, cmap, lim) in zip(axes[r], panels):
            im = ax.imshow(img, origin="lower", aspect="auto", cmap=cmap,
                           extent=[secs[0], secs[-1], 0, N_MELS],
                           vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
            yt = np.linspace(0, N_MELS - 1, 5).astype(int)
            ax.set_yticks(yt, [f"{hz[t]:.0f}" for t in yt])
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Class-averaged |SmoothGrad| saliency ({args.saliency_n} clips per class)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "saliency_class_mean.png")

    # ── Frequency and time profiles ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for name in DEEP:
        for c, ls in ((0, "--"), (1, "-")):
            m = class_mean[name][c]["SmoothGrad"]
            axes[0].plot(hz, m.sum(axis=1), ls, color=COLOURS[name], lw=2,
                         label=f"{name} — {CLASS_NAMES[c].lower()}")
            axes[1].plot(secs, m.sum(axis=0), ls, color=COLOURS[name], lw=1.5,
                         label=f"{name} — {CLASS_NAMES[c].lower()}")
    axes[0].set_xlabel("Frequency (Hz, mel-bin centre)")
    axes[0].set_ylabel("Share of attribution per mel bin")
    axes[0].set_title("Frequency profile (summed over time)", fontweight="bold")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Share of attribution per frame")
    axes[1].set_title("Time profile (summed over frequency)", fontweight="bold")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Where in time–frequency do the deep models look?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "saliency_profiles.png")

    # ── Deletion curves ───────────────────────────────────────────────────────
    deletion = pd.concat(deletion_frames, ignore_index=True)
    deletion.to_csv(RESULTS_DIR / "saliency_deletion_test.csv", index=False)
    styles = {"Vanilla gradient": ("#1E88E5", "-o"), "SmoothGrad": ("#43A047", "-s"),
              "Integrated Gradients": ("#E53935", "-^"), "Random": ("#757575", "--x")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, name in zip(axes, DEEP):
        d = deletion[deletion.model == name]
        for method, (col, fmt) in styles.items():
            dm = d[d.method == method]
            ax.plot(dm.fraction_deleted * 100, dm.mean_true_class_prob, fmt, color=col,
                    lw=2, label=method)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("% of time–frequency cells deleted (most salient first)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean true-class probability")
    axes[0].legend()
    fig.suptitle("Deletion test — salient regions vs random (steeper drop = more faithful)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "saliency_deletion_test.png")

    pd.DataFrame(freq_rows).to_csv(RESULTS_DIR / "saliency_band_shares.csv", index=False)
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "saliency_summary.csv", index=False)
    log("  Band attribution shares (SmoothGrad):\n" +
        pd.DataFrame(freq_rows).pivot_table(index=["model", "class"], columns="band_hz",
                                            values="attribution_share", sort=False)
        .to_string())
    log("  Deletion test @10%:\n" + pd.DataFrame(summary).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# Part C — Final evaluation & statistical testing
# ══════════════════════════════════════════════════════════════════════════════

def mcnemar_test(y, pred_a, pred_b):
    """McNemar's test on paired correctness. Exact binomial when discordant < 25."""
    a_ok, b_ok = pred_a == y, pred_b == y
    n01 = int((a_ok & ~b_ok).sum())    # A right, B wrong
    n10 = int((~a_ok & b_ok).sum())    # A wrong, B right
    n = n01 + n10
    if n == 0:
        return n01, n10, np.nan, 1.0, "none"
    if n < 25:
        return n01, n10, np.nan, stats.binomtest(n01, n, 0.5).pvalue, "exact"
    chi2 = (abs(n01 - n10) - 1) ** 2 / n
    return n01, n10, chi2, stats.chi2.sf(chi2, 1), "chi2-cc"


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n)
    out[order] = t + 1
    return out


def delong_test(y, s_a, s_b):
    """DeLong et al. (1988) test for two correlated ROC AUCs (Sun & Xu fast version)."""
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    preds = np.vstack([s_a, s_b])[:, order]
    n = preds.shape[1] - m
    tx = np.array([_midrank(p[:m]) for p in preds])
    ty = np.array([_midrank(p[m:]) for p in preds])
    tz = np.array([_midrank(p) for p in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], np.nan, 1.0 if aucs[0] == aucs[1] else np.nan
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    order = np.argsort(p[ok])
    pv = p[ok][order]
    m = len(pv)
    stepped = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * pv))
    tmp = np.empty(m)
    tmp[order] = stepped
    adj[ok] = tmp
    return adj


def stratified_bootstrap(y, n_boot, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def det_curve_points(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    fnr = 1 - tpr
    clip = lambda v: np.clip(v, 1e-4, 1 - 1e-4)
    return stats.norm.ppf(clip(fpr)), stats.norm.ppf(clip(fnr))


def p_heatmap(ax, mat, models, title):
    show = np.where(np.isnan(mat), 1.0, mat)
    ax.imshow(-np.log10(np.clip(show, 1e-12, 1)), cmap="Reds", vmin=0, vmax=6)
    ax.set_xticks(range(len(models)), models, rotation=30, ha="right")
    ax.set_yticks(range(len(models)), models)
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center")
                continue
            p = mat[i, j]
            txt = "n/a" if np.isnan(p) else ("<1e-6" if p < 1e-6 else
                                              f"{p:.0e}" if p < 1e-3 else f"{p:.3f}")
            star = "" if np.isnan(p) else "***" if p < 0.001 else "**" if p < 0.01 \
                else "*" if p < 0.05 else ""
            ax.text(j, i, f"{txt}\n{star}", ha="center", va="center", fontsize=8,
                    color="white" if not np.isnan(p) and p < 1e-4 else "black")
    ax.set_title(title, fontweight="bold", fontsize=10)


def run_stats(args):
    log("=" * 60)
    log(f"Part C — Final evaluation & statistical testing (B={args.bootstrap})")
    log("=" * 60)
    models = load_all_models()
    metric_rows, mcnemar_rows, delong_rows, eer_diff_rows = [], [], [], []
    preds_by_split = {}

    for split in SPLITS:
        X_hc, X_spec, y = load_split(split)
        preds = predict_all(models, X_hc, X_spec)
        preds_by_split[split] = (y, preds)
        rng = np.random.default_rng(SEED)

        # Bootstrap: same resamples for every model -> paired comparisons.
        boot = {m: {"eer": [], "auc": [], "f1": [], "acc": []} for m in ALL_MODELS}
        t0 = time.time()
        for b_idx in stratified_bootstrap(y, args.bootstrap, rng):
            yb = y[b_idx]
            for m, (pred, score) in preds.items():
                boot[m]["eer"].append(compute_eer(yb, score[b_idx]))
                boot[m]["auc"].append(roc_auc_score(yb, score[b_idx]))
                boot[m]["f1"].append(f1_score(yb, pred[b_idx]))
                boot[m]["acc"].append(accuracy_score(yb, pred[b_idx]))
        log(f"  [{split}] bootstrap done in {time.time() - t0:.0f}s")

        for m, (pred, score) in preds.items():
            met = compute_all_metrics(y, pred, score)
            row = {"model": m, "split": split, "n": len(y)}
            for key, bkey in [("eer", "eer"), ("roc_auc", "auc"), ("f1", "f1"),
                              ("accuracy", "acc")]:
                lo, hi = np.percentile(boot[m][bkey], [2.5, 97.5])
                row.update({key: met[key], f"{key}_ci_low": lo, f"{key}_ci_high": hi})
            row.update({"precision": met["precision"], "recall": met["recall"]})
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            row.update({"FP (genuine flagged)": fp, "FN (synthetic missed)": fn})
            metric_rows.append(row)

        pairs = list(combinations(ALL_MODELS, 2))
        mc, dl, ed = [], [], []
        for a, b in pairs:
            n01, n10, chi2, p, method = mcnemar_test(y, preds[a][0], preds[b][0])
            mc.append({"split": split, "A": a, "B": b, "A_right_B_wrong": n01,
                       "A_wrong_B_right": n10, "chi2": chi2, "p": p, "method": method})
            auc_a, auc_b, z, p = delong_test(y.astype(int), preds[a][1], preds[b][1])
            dl.append({"split": split, "A": a, "B": b, "AUC_A": auc_a, "AUC_B": auc_b,
                       "z": z, "p": p})
            diff = np.array(boot[a]["eer"]) - np.array(boot[b]["eer"])
            p_boot = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
            lo, hi = np.percentile(diff, [2.5, 97.5])
            ed.append({"split": split, "A": a, "B": b,
                       "EER_A": compute_eer(y, preds[a][1]),
                       "EER_B": compute_eer(y, preds[b][1]),
                       "dEER (A-B)": float(np.mean(diff)), "ci_low": lo, "ci_high": hi,
                       "p": p_boot})
        for rows, out in ((mc, mcnemar_rows), (dl, delong_rows), (ed, eer_diff_rows)):
            adj = holm([r["p"] for r in rows])
            for r, pa in zip(rows, adj):
                r["p_holm"] = pa
                r["significant_0.05"] = bool(pa < 0.05) if not np.isnan(pa) else False
            out.extend(rows)

    metrics = pd.DataFrame(metric_rows)
    metrics.round(5).to_csv(RESULTS_DIR / "final_metrics_with_ci.csv", index=False)
    pd.DataFrame(mcnemar_rows).to_csv(RESULTS_DIR / "mcnemar_tests.csv", index=False)
    pd.DataFrame(delong_rows).to_csv(RESULTS_DIR / "delong_auc_tests.csv", index=False)
    pd.DataFrame(eer_diff_rows).to_csv(RESULTS_DIR / "eer_bootstrap_tests.csv", index=False)

    # ── EER with 95% CI (forest plot) ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, split in zip(axes, SPLITS):
        d = metrics[metrics.split == split].set_index("model").loc[ALL_MODELS]
        ypos = np.arange(len(ALL_MODELS))[::-1]
        for yp, (m, r) in zip(ypos, d.iterrows()):
            ax.errorbar(r.eer * 100, yp,
                        xerr=[[(r.eer - r.eer_ci_low) * 100], [(r.eer_ci_high - r.eer) * 100]],
                        fmt="o", color=COLOURS[m], capsize=5, lw=2, ms=8)
            ax.text(r.eer_ci_high * 100, yp + 0.18,
                    f" {r.eer * 100:.2f}% [{r.eer_ci_low * 100:.2f}, {r.eer_ci_high * 100:.2f}]",
                    fontsize=8, va="bottom")
        ax.set_yticks(ypos, ALL_MODELS)
        ax.set_xlabel("EER (%) with 95% bootstrap CI")
        ax.set_title(SPLITS[split], fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
        ax.set_xlim(left=0, right=metrics.eer_ci_high.max() * 135)
    fig.suptitle("Equal Error Rate — point estimate and 95% CI", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "eer_confidence_intervals.png")

    # ── Significance heatmaps (Holm-adjusted p) ───────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for r, split in enumerate(SPLITS):
        for c, (rows, title) in enumerate([(mcnemar_rows, "McNemar (error patterns)"),
                                           (delong_rows, "DeLong (ROC AUC)"),
                                           (eer_diff_rows, "Paired bootstrap (EER)")]):
            mat = np.full((len(ALL_MODELS), len(ALL_MODELS)), np.nan)
            for row in rows:
                if row["split"] != split:
                    continue
                i, j = ALL_MODELS.index(row["A"]), ALL_MODELS.index(row["B"])
                mat[i, j] = mat[j, i] = row["p_holm"]
            p_heatmap(axes[r, c], mat, ALL_MODELS, f"{title}\n{SPLITS[split]}")
    fig.suptitle("Pairwise significance — Holm-adjusted p-values "
                 "(* p<0.05, ** p<0.01, *** p<0.001)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "significance_heatmaps.png")

    # ── DET curves ────────────────────────────────────────────────────────────
    ticks = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, split in zip(axes, SPLITS):
        y, preds = preds_by_split[split]
        for m in ALL_MODELS:
            x_d, y_d = det_curve_points(y, preds[m][1])
            ax.plot(x_d, y_d, color=COLOURS[m], lw=2,
                    label=f"{m} (EER={compute_eer(y, preds[m][1]) * 100:.2f}%)")
        tk = stats.norm.ppf(ticks)
        ax.plot(tk, tk, "k:", alpha=0.5, label="EER line")
        ax.set_xticks(tk, [f"{t * 100:g}" for t in ticks])
        ax.set_yticks(tk, [f"{t * 100:g}" for t in ticks])
        ax.set_xlim(tk[0], tk[-1])
        ax.set_ylim(tk[0], tk[-1])
        ax.set_xlabel("False acceptance rate — genuine flagged as synthetic (%)")
        ax.set_ylabel("Miss rate — synthetic passed as genuine (%)")
        ax.set_title(SPLITS[split], fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("DET curves (normal-deviate scale)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "det_curves.png")

    # ── Thesis-ready text summary ─────────────────────────────────────────────
    with open(RESULTS_DIR / "phase3_final_evaluation.txt", "w", encoding="utf-8") as f:
        f.write("Phase 3 — Final Evaluation with Statistical Testing\n")
        f.write("=" * 78 + "\n")
        f.write(f"Bootstrap resamples: {args.bootstrap} (stratified); seed {SEED}\n")
        f.write("Thresholds: classical = validation-EER threshold (Phase 2); "
                "deep = argmax (P>=0.5)\n\n")
        for split in SPLITS:
            f.write(f"{SPLITS[split]}\n" + "-" * 78 + "\n")
            d = metrics[metrics.split == split]
            for _, r in d.iterrows():
                f.write(f"  {r.model:18s} EER {r.eer * 100:6.2f}% "
                        f"[{r.eer_ci_low * 100:5.2f}, {r.eer_ci_high * 100:5.2f}]   "
                        f"AUC {r.roc_auc:.4f} [{r.roc_auc_ci_low:.4f}, {r.roc_auc_ci_high:.4f}]   "
                        f"F1 {r.f1:.4f}   FP {r['FP (genuine flagged)']:3d}  "
                        f"FN {r['FN (synthetic missed)']:3d}\n")
            f.write("\n  Pairwise tests (Holm-adjusted p):\n")
            f.write(f"  {'A':18s} {'B':18s} {'McNemar':>9s} {'DeLong':>9s} {'ΔEER boot':>10s}"
                    f"   ΔEER (A−B) [95% CI]\n")
            for mc, dl, ed in zip(mcnemar_rows, delong_rows, eer_diff_rows):
                if mc["split"] != split:
                    continue
                fmt = lambda p: "     n/a" if np.isnan(p) else f"{p:9.4f}"
                f.write(f"  {mc['A']:18s} {mc['B']:18s} {fmt(mc['p_holm'])} "
                        f"{fmt(dl['p_holm'])} {fmt(ed['p_holm']):>10s}   "
                        f"{ed['dEER (A-B)'] * 100:+.2f} pp "
                        f"[{ed['ci_low'] * 100:+.2f}, {ed['ci_high'] * 100:+.2f}]\n")
            f.write("\n")
    log("  Saved: results/phase3_final_evaluation.txt")
    print((RESULTS_DIR / "phase3_final_evaluation.txt").read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
# Part D — Robustness of all five models (noise + G.711)
# ══════════════════════════════════════════════════════════════════════════════

CONDITIONS = [  # (label, snr_db or None, apply_g711)
    ("Clean", None, False),
    ("20 dB", 20, False),
    ("15 dB", 15, False),
    ("10 dB", 10, False),
    ("5 dB", 5, False),
    ("G.711", None, True),
    ("G.711 + 10 dB", 10, True),
]


def resolve_audio_path(p):
    """Manifests store absolute Windows paths; remap them when run elsewhere (Colab)."""
    path = Path(p)
    if path.exists():
        return path
    parts = Path(p.replace("\\", "/")).parts
    if "voiceauth_project" in parts:
        cand = ROOT.joinpath(*parts[parts.index("voiceauth_project") + 1:])
        if cand.exists():
            return cand
    raise FileNotFoundError(p)


def _features_for_clip(path, clip_seed, with_handcrafted):
    from src.utils.audio_utils import load_audio, add_white_noise, simulate_g711
    from src.features.extractor import (extract_all_handcrafted,
                                        extract_log_mel_spectrogram, normalize_spectrogram)
    audio, sr = load_audio(str(path), target_sr=TARGET_SR)
    hc_out, spec_out = [], []
    for c, (_, snr, codec) in enumerate(CONDITIONS):
        a = audio.copy()
        if snr is not None:
            np.random.seed(clip_seed * 100 + c)   # same noise for every model
            a = add_white_noise(a, snr_db=snr)
        if codec:
            a = simulate_g711(a, sr=TARGET_SR)
        a = a.astype(np.float32)
        spec = normalize_spectrogram(extract_log_mel_spectrogram(
            a, sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP))
        spec_out.append(spec[np.newaxis])
        hc_out.append(extract_all_handcrafted(a, sr, n_mfcc=N_MFCC) if with_handcrafted
                      else np.zeros(1))
    return np.stack(hc_out), np.stack(spec_out)


def run_robustness(args):
    from joblib import Parallel, delayed

    log("=" * 60)
    log(f"Part D — Robustness of all five models ({args.robust_n} test clips)")
    log("=" * 60)
    with open(PROC_DIR / "test_manifest.json") as f:
        manifest = json.load(f)
    y_all = np.array([m["label"] for m in manifest])
    idx = balanced_sample(y_all, args.robust_n, np.random.default_rng(SEED))
    y = y_all[idx]

    cache = CACHE_DIR / f"robustness_features_n{len(idx)}.npz"
    if cache.exists():
        data = np.load(cache)
        X_hc, X_spec = data["X_hc"], data["X_spec"]
        log(f"  Loaded cached features: {cache.name}")
    else:
        paths = [resolve_audio_path(manifest[i]["path"]) for i in idx]
        t0 = time.time()
        # Compile librosa's numba kernels (pyin) once here: many workers compiling
        # and writing the numba cache concurrently crashes on Windows.
        _features_for_clip(paths[0], int(idx[0]), True)
        n_jobs = args.jobs if args.jobs > 0 else max(1, min(8, (os.cpu_count() or 2) - 1))
        log(f"  Extracting features with {n_jobs} workers...")
        res = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_features_for_clip)(p, int(i), True) for p, i in zip(paths, idx))
        X_hc = np.stack([r[0] for r in res], axis=1).astype(np.float32)    # (C, N, 257)
        X_spec = np.stack([r[1] for r in res], axis=1).astype(np.float32)  # (C, N, 1, 128, 128)
        np.savez_compressed(cache, X_hc=X_hc, X_spec=X_spec, y=y, idx=idx)
        log(f"  Extracted {len(CONDITIONS)} conditions × {len(idx)} clips "
            f"in {time.time() - t0:.0f}s (cached to {cache.name})")

    models = load_all_models()
    rows = []
    for c, (label, _, _) in enumerate(CONDITIONS):
        preds = predict_all(models, np.nan_to_num(X_hc[c]), X_spec[c])
        for m, (pred, score) in preds.items():
            met = compute_all_metrics(y, pred, score)
            rows.append({"condition": label, "model": m, "eer": met["eer"],
                         "roc_auc": met["roc_auc"], "accuracy": met["accuracy"],
                         "FAR (genuine flagged)": float(((pred == 1) & (y == 0)).sum() /
                                                        (y == 0).sum()),
                         "FRR (synthetic missed)": float(((pred == 0) & (y == 1)).sum() /
                                                         (y == 1).sum())})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(RESULTS_DIR / "robustness_all_models.csv", index=False)

    labels = [c[0] for c in CONDITIONS]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for m in ALL_MODELS:
        d = df[df.model == m].set_index("condition").loc[labels]
        lw = 2.5 if m in DEEP else 1.8
        ls = "-" if m in DEEP else "--"
        axes[0].plot(labels, d.eer * 100, ls, marker="o", color=COLOURS[m], lw=lw, label=m)
        axes[1].plot(labels, d["FAR (genuine flagged)"] * 100, ls, marker="o",
                     color=COLOURS[m], lw=lw, label=m)
        axes[2].plot(labels, d["FRR (synthetic missed)"] * 100, ls, marker="o",
                     color=COLOURS[m], lw=lw, label=m)
    for ax, t in zip(axes, ["EER (threshold-free)",
                            "Genuine callers flagged at deployed threshold",
                            "Synthetic speech missed at deployed threshold"]):
        ax.set_title(t, fontweight="bold")
        ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
        ax.axvline(4.5, color="k", lw=0.8, alpha=0.4)
    axes[0].legend()
    fig.suptitle(f"Robustness — all five models under noise and G.711 "
                 f"({len(idx)} balanced test clips; white noise)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "robustness_all_models.png")

    log("  EER (%) by condition:\n" +
        (df.pivot(index="model", columns="condition", values="eer")
         .loc[ALL_MODELS, labels] * 100).round(2).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

PARTS = {"shap": run_shap, "saliency": run_saliency, "stats": run_stats,
         "robustness": run_robustness}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 3: interpretability + final evaluation")
    ap.add_argument("--parts", nargs="+", choices=list(PARTS), default=list(PARTS))
    ap.add_argument("--shap-n", type=int, default=300,
                    help="test clips explained with TreeSHAP (RF, GB)")
    ap.add_argument("--shap-svm-n", type=int, default=150,
                    help="test clips explained with permutation SHAP (SVM)")
    ap.add_argument("--shap-background", type=int, default=50,
                    help="training clips used as SVM SHAP background")
    ap.add_argument("--saliency-n", type=int, default=100,
                    help="correctly classified test clips per class for saliency")
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--smoothgrad-n", type=int, default=16)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--robust-n", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=-1,
                    help="parallel workers for robustness (-1 = auto, capped at 8)")
    ap.add_argument("--quick", action="store_true", help="tiny samples for a smoke test")
    args = ap.parse_args()
    if args.quick:
        args.shap_n, args.shap_svm_n, args.shap_background = 20, 6, 10
        args.saliency_n, args.ig_steps, args.smoothgrad_n = 4, 4, 2
        args.bootstrap, args.robust_n = 20, 10

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    log(f"Phase 3 analysis — parts: {', '.join(args.parts)} | device: {DEVICE}")
    t_start = time.time()
    for part in args.parts:
        PARTS[part](args)
    log(f"Done in {(time.time() - t_start) / 60:.1f} min. Outputs: {OUT_DIR}")
