"""
scripts/02_extract_features.py  [v2]
Step 2: Extract handcrafted features and log-Mel spectrograms.

Changes from v1:
  - Also processes generalization_test split if manifest exists
  - Progress bar shows ETA (useful for 12k+ clips)
  - Skips splits whose .npy files already exist (resume-friendly)

Run: python scripts/02_extract_features.py
     python scripts/02_extract_features.py --splits train  (single split)
     python scripts/02_extract_features.py --force         (re-extract everything)
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.features.extractor import (
    extract_all_handcrafted, extract_log_mel_spectrogram,
    normalize_spectrogram, get_feature_names
)
from src.utils.audio_utils import load_audio

TARGET_SR = cfg["audio"]["target_sr"]
N_MFCC    = cfg["features"]["n_mfcc"]
N_MELS    = cfg["features"]["n_mels"]
N_FFT     = cfg["features"]["n_fft"]
HOP       = cfg["features"]["hop_length"]
PROC_DIR  = ROOT / cfg["paths"]["data_processed"]


def process_split(split_name: str, force: bool = False):
    manifest_path = PROC_DIR / f"{split_name}_manifest.json"
    hc_path   = PROC_DIR / f"X_handcrafted_{split_name}.npy"
    spec_path = PROC_DIR / f"X_spectrogram_{split_name}.npy"
    y_path    = PROC_DIR / f"y_{split_name}.npy"

    if not manifest_path.exists():
        print(f"[{split_name}] No manifest found — skipping.")
        return

    if not force and hc_path.exists() and spec_path.exists() and y_path.exists():
        y = np.load(y_path)
        print(f"[{split_name}] Already extracted ({len(y)} clips) — skipping. Use --force to redo.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\n[{split_name}] Extracting features from {len(manifest)} clips...")

    handcrafted_list, spectrogram_list, label_list, failed = [], [], [], []

    for item in tqdm(manifest, desc=split_name):
        try:
            audio, sr = load_audio(item["path"], target_sr=TARGET_SR)

            hc   = extract_all_handcrafted(audio, sr, n_mfcc=N_MFCC)
            spec = extract_log_mel_spectrogram(audio, sr, n_mels=N_MELS,
                                                n_fft=N_FFT, hop_length=HOP)
            spec = normalize_spectrogram(spec)

            handcrafted_list.append(hc)
            spectrogram_list.append(spec)
            label_list.append(item["label"])

        except Exception as e:
            failed.append((item["path"], str(e)))

    if failed:
        print(f"  [!] {len(failed)} files failed (first 3):")
        for path, err in failed[:3]:
            print(f"      {Path(path).name}: {err}")

    X_hc   = np.array(handcrafted_list, dtype=np.float32)
    X_spec = np.array(spectrogram_list, dtype=np.float32)[:, np.newaxis, :, :]
    y      = np.array(label_list, dtype=np.int64)

    np.save(hc_path,   X_hc)
    np.save(spec_path, X_spec)
    np.save(y_path,    y)

    ng, ns = (y == 0).sum(), (y == 1).sum()
    print(f"  Handcrafted : {X_hc.shape}  |  Spectrogram: {X_spec.shape}")
    print(f"  Labels      : genuine={ng}  synthetic={ns}")
    print(f"  Saved to    : {PROC_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+",
                        default=["train", "val", "test", "generalization_test"],
                        help="Which splits to process")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if .npy files already exist")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 2: Feature Extraction")
    print("=" * 60)

    names = get_feature_names(N_MFCC)
    import json as _json
    with open(PROC_DIR / "feature_names.json", "w") as f:
        _json.dump(names, f, indent=2)
    print(f"Feature vector length: {len(names)}")

    for split in args.splits:
        process_split(split, force=args.force)

    print("\n✓ Feature extraction complete.")
    print("  Next: python scripts/03_train_baselines.py")