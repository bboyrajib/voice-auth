"""
scripts/04_train_cnn.py  [v2 — dataset-scale aware]
Step 4: Train CNN and CNN-LSTM on log-Mel spectrograms.

Changes from v1:
  - Disk-based Dataset (no full .npy load into RAM — 12k spectrograms = ~6GB)
  - Weighted sampler replaces SMOTE for deep models (handles 0.27:1 imbalance)
  - Also evaluates on generalization_test (edge_tts held-out)
  - Windows-safe: num_workers=0 by default (set --workers 2 on Linux/Colab)
  - Mixed precision training (faster on RTX 3060, automatic on Colab A100)

Run:
  python scripts/04_train_cnn.py --model cnn           # ~2h on RTX 3060
  python scripts/04_train_cnn.py --model cnn_lstm      # ~3h on RTX 3060
  python scripts/04_train_cnn.py --model cnn --workers 0  # Windows safe
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.models.cnn import SpeechCNN, SpeechCNNLSTM
from src.evaluation.metrics import compute_all_metrics

PROC_DIR    = ROOT / cfg["paths"]["data_processed"]
MODELS_DIR  = ROOT / cfg["paths"]["outputs_models"]
RESULTS_DIR = ROOT / cfg["paths"]["outputs_results"]
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = cfg["project"]["seed"]
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Disk-based Dataset ─────────────────────────────────────────────────────────
# Loads spectrograms one batch at a time instead of the full array.
# With 12k × (1×128×128) float32 = ~6.4GB — too large to hold in RAM safely.

class SpectrogramDataset(Dataset):
    """
    Memory-mapped spectrogram dataset.
    Reads directly from .npy files without loading all into RAM.
    """
    def __init__(self, split: str, proc_dir: Path):
        # Use mmap_mode='r' — reads slices from disk on demand
        self.X = np.load(proc_dir / f"X_spectrogram_{split}.npy", mmap_mode='r')
        self.y = np.load(proc_dir / f"y_{split}.npy")
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Copy the slice to avoid mmap reference issues across workers
        x = torch.tensor(self.X[idx].copy(), dtype=torch.float32)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y

    def get_labels(self):
        return self.y.tolist()


def make_weighted_sampler(labels):
    """
    WeightedRandomSampler: each class gets equal expected representation per batch.
    This is the deep learning equivalent of class_weight='balanced'.
    """
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(labels),
        replacement=True
    )


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name}  ({mem:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("No GPU — using CPU (will be slow, consider Colab)")
    return device


# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

    metrics = compute_all_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    metrics["loss"] = total_loss / len(all_labels)
    return metrics


# ── Main training function ─────────────────────────────────────────────────────

def train_model(model_name: str, num_workers: int = 0, batch_override: int = None):
    model_cfg = cfg["models"]["deep"][model_name.replace("-", "_").lower()]
    BATCH    = batch_override if batch_override else model_cfg["batch_size"]
    EPOCHS   = model_cfg["epochs"]
    LR       = model_cfg["lr"]
    PATIENCE = model_cfg["patience"]

    device = get_device()

    # Auto-adjust batch size based on available VRAM
    if device.type == 'cuda' and batch_override is None:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 7:      # 6GB VRAM (RTX 3060 Laptop)
            BATCH = 16
            print(f'  6GB VRAM detected — auto-setting batch size to {BATCH}')
        elif vram_gb < 13:   # 8-12GB
            BATCH = 32
            print(f'  {vram_gb:.0f}GB VRAM detected — batch size {BATCH}')
        else:                # 16GB+ (A100 etc)
            BATCH = 64
            print(f'  {vram_gb:.0f}GB VRAM detected — batch size {BATCH}')

    # Datasets
    print("\nLoading datasets (memory-mapped)...")
    train_ds = SpectrogramDataset("train", PROC_DIR)
    val_ds   = SpectrogramDataset("val",   PROC_DIR)
    test_ds  = SpectrogramDataset("test",  PROC_DIR)

    print(f"  Train: {len(train_ds):6d}  Val: {len(val_ds):6d}  Test: {len(test_ds):6d}")

    # Weighted sampler on train to handle 0.27:1 imbalance
    sampler = make_weighted_sampler(train_ds.get_labels())

    # num_workers=0 is safest on Windows; use 2-4 on Linux/Colab
    loader_kwargs = dict(batch_size=BATCH, num_workers=num_workers,
                         pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    val_loader   = DataLoader(val_ds,  shuffle=False,    **loader_kwargs)
    test_loader  = DataLoader(test_ds, shuffle=False,    **loader_kwargs)

    # Model
    if model_name == "cnn":
        model = SpeechCNN().to(device)
    elif model_name == "cnn_lstm":
        model = SpeechCNNLSTM().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_name.upper()} | Parameters: {n_params:,}")

    # Weighted cross-entropy as a secondary defense against imbalance
    labels     = np.array(train_ds.get_labels())
    counts     = np.bincount(labels)
    cw         = torch.tensor(len(labels) / (2.0 * counts), dtype=torch.float32).to(device)
    criterion  = nn.CrossEntropyLoss(weight=cw)

    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler     = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_eer       = float("inf")
    best_epoch     = 0
    patience_count = 0
    history        = []

    print(f"\nTraining up to {EPOCHS} epochs  (early stop patience={PATIENCE})")
    print(f"{'Ep':>4} {'TrainLoss':>10} {'TrainAcc':>9} {'ValEER':>8} {'ValF1':>8} {'ValAUC':>8}")
    print("─" * 55)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                             criterion, device, scaler)
        val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc,  5),
            **{f"val_{k}": round(v, 5) for k, v in val_m.items()}
        })

        print(f"{epoch:>4} {train_loss:>10.4f} {train_acc:>9.4f} "
              f"{val_m['eer']:>8.4f} {val_m['f1']:>8.4f} {val_m['roc_auc']:>8.4f}")

        if val_m["eer"] < best_eer:
            best_eer   = val_m["eer"]
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), MODELS_DIR / f"{model_name}_best.pt")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stop at epoch {epoch}. "
                      f"Best: epoch {best_epoch}  EER={best_eer:.4f}")
                break

    # ── Evaluate best model on test + generalization_test ─────────────────────
    print(f"\nLoading best model (epoch {best_epoch})...")
    model.load_state_dict(torch.load(MODELS_DIR / f"{model_name}_best.pt",
                                      map_location=device))

    test_m = evaluate(model, test_loader, criterion, device)
    print(f"\n[Test]        EER={test_m['eer']:.4f} | F1={test_m['f1']:.4f} | AUC={test_m['roc_auc']:.4f}")

    all_results = [{"model": model_name, "split": "test", **test_m}]

    # Generalization test (edge_tts held-out) if available
    gen_manifest = PROC_DIR / "generalization_test_manifest.json"
    if gen_manifest.exists():
        # Build a quick npy-less dataset from manifest paths directly
        gen_ds = SpectrogramDataset("generalization_test", PROC_DIR) \
            if (PROC_DIR / "X_spectrogram_generalization_test.npy").exists() \
            else None

        if gen_ds:
            gen_loader = DataLoader(gen_ds, shuffle=False, **loader_kwargs)
            gen_m = evaluate(model, gen_loader, criterion, device)
            print(f"[GenTest/EdgeTTS] EER={gen_m['eer']:.4f} | F1={gen_m['f1']:.4f} | AUC={gen_m['roc_auc']:.4f}")
            all_results.append({"model": model_name, "split": "generalization_test", **gen_m})
        else:
            print("[GenTest] Run 02_extract_features.py with generalization_test split first.")

    # Save
    pd.DataFrame(history).to_csv(RESULTS_DIR / f"{model_name}_history.csv", index=False)
    pd.DataFrame(all_results).to_csv(RESULTS_DIR / f"{model_name}_test_results.csv", index=False)

    print(f"\n✓ {model_name.upper()} complete.")
    return test_m


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["cnn", "cnn_lstm"], default="cnn")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers. Use 0 on Windows, 2-4 on Linux/Colab")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size (default: from config). Use 16 for 6GB VRAM, 32 for 12GB")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Step 4: Deep Learning — {args.model.upper()}")
    print("=" * 60)

    train_model(args.model, num_workers=args.workers, batch_override=args.batch)
    print("\nNext: python scripts/04_train_cnn.py --model cnn_lstm")
    print("Then: python scripts/05_evaluate.py")
