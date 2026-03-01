"""
scripts/01_download_data.py
Step 1: Download genuine speech datasets and generate synthetic speech.

GENUINE SPEECH:
  - LibriSpeech test-clean (small subset, ~1GB, easy to download)
  - We use test-clean to avoid train/val contamination with pretrained models

SYNTHETIC SPEECH:
  - Generated using Coqui TTS (open-source, runs locally)
  - Models: Tacotron2, VITS — covers two generation paradigms
  - Phrases simulating banking authentication utterances

Run: python scripts/01_download_data.py
"""

import os
import sys
import subprocess
import urllib.request
import tarfile
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Add project root to path ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.audio_utils import load_audio, save_audio, trim_or_pad
import yaml

with open(ROOT / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

random.seed(cfg["project"]["seed"])
np.random.seed(cfg["project"]["seed"])

GENUINE_DIR = ROOT / cfg["paths"]["data_raw_genuine"]
SYNTHETIC_DIR = ROOT / cfg["paths"]["data_raw_synthetic"]
GENUINE_DIR.mkdir(parents=True, exist_ok=True)
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = cfg["audio"]["target_sr"]
MIN_DUR = cfg["audio"]["min_duration"]
MAX_DUR = cfg["audio"]["max_duration"]


# ── Banking-style phrases for TTS synthesis ────────────────────────────────────
# These simulate real transaction confirmation utterances

BANKING_PHRASES = [
    "Yes, I confirm this transaction.",
    "Approve the transfer of five hundred dollars.",
    "I authorize this payment.",
    "Please proceed with the withdrawal.",
    "Yes, that is correct, go ahead.",
    "I confirm my identity, Rajib Roy.",
    "Authorize the transfer to my savings account.",
    "Yes, I want to send money to account ending in four seven two one.",
    "I approve this transaction of two thousand rupees.",
    "Please verify my account and proceed.",
    "Yes, confirm the transfer.",
    "I consent to this transaction.",
    "Proceed with the payment.",
    "Transfer approved.",
    "Yes, that's my account, go ahead with it.",
    "I confirm. Please proceed.",
    "Authorization granted for this transfer.",
    "I want to approve the payment now.",
    "Yes, send it.",
    "Confirm the transaction please.",
]


# ── Step 1a: Download LibriSpeech test-clean ───────────────────────────────────

def download_librispeech():
    """Download LibriSpeech test-clean (~400MB) — 40 speakers, clean read speech."""
    url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    archive_path = GENUINE_DIR / "test-clean.tar.gz"
    extract_path = GENUINE_DIR / "LibriSpeech"

    if extract_path.exists():
        print("[LibriSpeech] Already downloaded. Skipping.")
        return

    print(f"[LibriSpeech] Downloading test-clean (~400MB)...")
    print(f"  URL: {url}")
    print("  This may take a few minutes depending on your connection.")

    class ProgressBar:
        def __init__(self):
            self.pbar = None
        def __call__(self, block_num, block_size, total_size):
            if self.pbar is None:
                self.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
            self.pbar.update(block_size)

    urllib.request.urlretrieve(url, archive_path, ProgressBar())
    print("\n[LibriSpeech] Extracting...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(GENUINE_DIR)
    archive_path.unlink()  # Remove archive to save space
    print("[LibriSpeech] Done.")


def prepare_genuine_clips():
    """
    Process LibriSpeech flac files:
    - Resample to 8kHz
    - Trim/pad to 3-8 seconds
    - Save as wav with speaker metadata
    """
    libri_dir = GENUINE_DIR / "LibriSpeech" / "test-clean"
    if not libri_dir.exists():
        print("[Genuine] LibriSpeech not found. Run download first.")
        return

    processed_dir = GENUINE_DIR / "processed"
    processed_dir.mkdir(exist_ok=True)

    manifest = []
    all_files = list(libri_dir.rglob("*.flac"))
    print(f"[Genuine] Processing {len(all_files)} LibriSpeech files...")

    skipped = 0
    for fpath in tqdm(all_files):
        speaker_id = fpath.parts[-3]  # LibriSpeech folder structure: speaker/chapter/file
        audio, sr = load_audio(str(fpath), target_sr=TARGET_SR)
        audio = trim_or_pad(audio, sr, MIN_DUR, MAX_DUR)

        if audio is None:
            skipped += 1
            continue

        out_name = f"genuine_{speaker_id}_{fpath.stem}.wav"
        out_path = processed_dir / out_name
        save_audio(audio, str(out_path), sr=TARGET_SR)

        manifest.append({
            "path": str(out_path),
            "label": 0,           # 0 = genuine
            "label_str": "genuine",
            "speaker_id": speaker_id,
            "source": "librispeech",
            "duration": len(audio) / TARGET_SR
        })

    print(f"[Genuine] Processed {len(manifest)} clips. Skipped {skipped} (too short).")

    # Save manifest
    manifest_path = GENUINE_DIR / "genuine_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Genuine] Manifest saved to {manifest_path}")
    return manifest


# ── Step 1b: Generate synthetic speech with Coqui TTS ─────────────────────────

def check_coqui_tts():
    """Check if Coqui TTS is installed; install if not."""
    try:
        import TTS
        print("[TTS] Coqui TTS is installed.")
        return True
    except ImportError:
        print("[TTS] Coqui TTS not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "TTS", "--quiet"], check=True)
        print("[TTS] Installed.")
        return True


def generate_synthetic_clips():
    """
    Generate AI speech using two TTS models:
    - tts_models/en/ljspeech/tacotron2-DDC  (Tacotron2-based)
    - tts_models/en/vctk/vits               (VITS-based, multi-speaker)
    Each phrase is synthesized with both models.
    """
    check_coqui_tts()
    from TTS.api import TTS as CoquiTTS

    out_dir = SYNTHETIC_DIR / "processed"
    out_dir.mkdir(exist_ok=True)

    manifest = []

    # Model 1: Tacotron2
    print("\n[Synthetic] Generating with Tacotron2-DDC...")
    try:
        tts_t2 = CoquiTTS("tts_models/en/ljspeech/tacotron2-DDC")
        for i, phrase in enumerate(tqdm(BANKING_PHRASES, desc="Tacotron2")):
            out_path = out_dir / f"tacotron2_{i:03d}.wav"
            tts_t2.tts_to_file(text=phrase, file_path=str(out_path))

            # Load, resample, trim/pad
            audio, sr = load_audio(str(out_path), target_sr=TARGET_SR)
            audio = trim_or_pad(audio, TARGET_SR, MIN_DUR, MAX_DUR)
            if audio is None:
                continue
            save_audio(audio, str(out_path), sr=TARGET_SR)

            manifest.append({
                "path": str(out_path),
                "label": 1,
                "label_str": "synthetic",
                "speaker_id": "tacotron2_ljspeech",
                "source": "tacotron2",
                "phrase": phrase,
                "duration": len(audio) / TARGET_SR
            })
    except Exception as e:
        print(f"[Tacotron2] Error: {e}. Skipping this model.")

    # Model 2: VITS (multi-speaker)
    print("\n[Synthetic] Generating with VITS...")
    try:
        tts_vits = CoquiTTS("tts_models/en/vctk/vits")
        vits_speakers = ["p225", "p226", "p227", "p228", "p229"]  # Different VCTK speakers

        for spk in vits_speakers:
            for i, phrase in enumerate(tqdm(BANKING_PHRASES, desc=f"VITS-{spk}")):
                out_path = out_dir / f"vits_{spk}_{i:03d}.wav"
                tts_vits.tts_to_file(text=phrase, file_path=str(out_path), speaker=spk)

                audio, sr = load_audio(str(out_path), target_sr=TARGET_SR)
                audio = trim_or_pad(audio, TARGET_SR, MIN_DUR, MAX_DUR)
                if audio is None:
                    continue
                save_audio(audio, str(out_path), sr=TARGET_SR)

                manifest.append({
                    "path": str(out_path),
                    "label": 1,
                    "label_str": "synthetic",
                    "speaker_id": f"vits_{spk}",
                    "source": "vits",
                    "phrase": phrase,
                    "duration": len(audio) / TARGET_SR
                })
    except Exception as e:
        print(f"[VITS] Error: {e}. Skipping this model.")

    manifest_path = SYNTHETIC_DIR / "synthetic_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[Synthetic] Generated {len(manifest)} clips.")
    print(f"[Synthetic] Manifest saved to {manifest_path}")
    return manifest


# ── Step 1c: Build combined dataset split ─────────────────────────────────────

def build_splits():
    """
    Merge genuine and synthetic manifests.
    Split into train/val/test ensuring speaker independence.
    Save split manifests for downstream scripts.
    """
    import json
    import random

    genuine_manifest_path = GENUINE_DIR / "genuine_manifest.json"
    synthetic_manifest_path = SYNTHETIC_DIR / "synthetic_manifest.json"

    if not genuine_manifest_path.exists() or not synthetic_manifest_path.exists():
        print("[Splits] Manifests not found. Run download/generate steps first.")
        return

    with open(genuine_manifest_path) as f:
        genuine = json.load(f)
    with open(synthetic_manifest_path) as f:
        synthetic = json.load(f)

    # Speaker-independent split for genuine (synthetic has no real speaker to split on)
    genuine_speakers = list(set(c["speaker_id"] for c in genuine))
    random.shuffle(genuine_speakers)

    n = len(genuine_speakers)
    train_spk = set(genuine_speakers[:int(n * 0.70)])
    val_spk = set(genuine_speakers[int(n * 0.70):int(n * 0.85)])
    test_spk = set(genuine_speakers[int(n * 0.85):])

    train_genuine = [c for c in genuine if c["speaker_id"] in train_spk]
    val_genuine   = [c for c in genuine if c["speaker_id"] in val_spk]
    test_genuine  = [c for c in genuine if c["speaker_id"] in test_spk]

    # Split synthetic — use multiple sources if available, else split randomly
    sources = list(set(c["source"] for c in synthetic))
    if len(sources) >= 2:
        train_synthetic = [c for c in synthetic if c["source"] == sources[0]]
        remaining = [c for c in synthetic if c["source"] != sources[0]]
        mid = len(remaining) // 2
        val_synthetic  = remaining[:mid]
        test_synthetic = remaining[mid:]
    else:
        random.shuffle(synthetic)
        n_s = len(synthetic)
        train_synthetic = synthetic[:int(n_s * 0.70)]
        val_synthetic   = synthetic[int(n_s * 0.70):int(n_s * 0.85)]
        test_synthetic  = synthetic[int(n_s * 0.85):]
        print(f"[Splits] Only one TTS source ({sources[0]}). Splitting synthetic randomly.")

    splits = {
        "train": train_genuine + train_synthetic,
        "val":   val_genuine   + val_synthetic,
        "test":  test_genuine  + test_synthetic,
    }

    processed_root = ROOT / cfg["paths"]["data_processed"]
    processed_root.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    for split_name, items in splits.items():
        random.shuffle(items)
        out_path = processed_root / f"{split_name}_manifest.json"
        with open(out_path, "w") as f:
            json.dump(items, f, indent=2)

        n_genuine   = sum(1 for c in items if c["label"] == 0)
        n_synthetic = sum(1 for c in items if c["label"] == 1)
        print(f"[Splits] {split_name:6s}: {len(items):4d} total | {n_genuine} genuine | {n_synthetic} synthetic")

    print("\n[Splits] Done. Manifests saved to data/processed/")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Dataset Download and Preparation")
    print("=" * 60)

    print("\n--- Step 1a: Genuine Speech (LibriSpeech) ---")
    download_librispeech()
    prepare_genuine_clips()

    print("\n--- Step 1b: Synthetic Speech (Coqui TTS) ---")
    generate_synthetic_clips()

    print("\n--- Step 1c: Build Train/Val/Test Splits ---")
    build_splits()

    print("\n✓ Step 1 complete. Run scripts/02_extract_features.py next.")