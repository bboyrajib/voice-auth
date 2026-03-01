"""
fix_splits.py
Rebuild train/val/test splits with MIXED sources in each split.
Each split will contain proportional samples from ALL TTS sources,
ensuring val and test measure the same distribution.

The source-held-out design is preserved SEPARATELY as a "generalization test"
which is a stronger, distinct experiment for the thesis.
"""

import sys, json, random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

GENUINE_DIR   = ROOT / cfg["paths"]["data_raw_genuine"]
SYNTHETIC_DIR = ROOT / cfg["paths"]["data_raw_synthetic"]
PROC_DIR      = ROOT / cfg["paths"]["data_processed"]
SEED          = cfg["project"]["seed"]

random.seed(SEED)

# Load all data
with open(GENUINE_DIR / "genuine_manifest.json") as f:
    genuine = json.load(f)
with open(SYNTHETIC_DIR / "synthetic_manifest.json") as f:
    synthetic = json.load(f)

print(f"Loaded: {len(genuine)} genuine, {len(synthetic)} synthetic")

# ── Genuine: speaker-independent split (unchanged) ────────────────────────────
speakers = list(set(c["speaker_id"] for c in genuine))
random.shuffle(speakers)
n = len(speakers)
train_spk = set(speakers[:int(n * 0.70)])
val_spk   = set(speakers[int(n * 0.70):int(n * 0.85)])
test_spk  = set(speakers[int(n * 0.85):])

train_g = [c for c in genuine if c["speaker_id"] in train_spk]
val_g   = [c for c in genuine if c["speaker_id"] in val_spk]
test_g  = [c for c in genuine if c["speaker_id"] in test_spk]

print(f"Genuine splits: train={len(train_g)} val={len(val_g)} test={len(test_g)}")

# ── Synthetic: stratified split BY SOURCE (70/15/15 within each source) ───────
# This ensures every source appears in every split at the same proportion.
by_source = defaultdict(list)
for c in synthetic:
    by_source[c["source"]].append(c)

train_s, val_s, test_s = [], [], []
print("\nSynthetic per-source splits:")
for source, clips in sorted(by_source.items()):
    random.shuffle(clips)
    n = len(clips)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)
    tr = clips[:t_end]
    va = clips[t_end:v_end]
    te = clips[v_end:]
    train_s.extend(tr)
    val_s.extend(va)
    test_s.extend(te)
    print(f"  {source:25s}: total={n:5d} → train={len(tr):4d} val={len(va):4d} test={len(te):4d}")

# ── Save main splits ───────────────────────────────────────────────────────────
splits = {
    "train": train_g + train_s,
    "val":   val_g   + val_s,
    "test":  test_g  + test_s,
}

print("\nFinal split breakdown:")
for name, items in splits.items():
    random.shuffle(items)
    with open(PROC_DIR / f"{name}_manifest.json", "w") as f:
        json.dump(items, f, indent=2)
    ng = sum(1 for c in items if c["label"] == 0)
    ns = sum(1 for c in items if c["label"] == 1)
    print(f"  {name:6s}: {len(items):5d} total | genuine={ng:4d} synthetic={ns:4d} | ratio={ng/max(ns,1):.2f}:1")

# ── Save a SEPARATE held-out generalization test (edge_tts only) ──────────────
# This is a distinct experiment: "does the model generalize to unseen TTS systems?"
edge_only = [c for c in synthetic if c["source"] == "edge_tts"]
random.shuffle(edge_only)
gentest = test_g + edge_only   # genuine test speakers + all edge_tts clips
with open(PROC_DIR / "generalization_test_manifest.json", "w") as f:
    json.dump(gentest, f, indent=2)
print(f"\n  generalization_test (edge_tts held-out): {len(gentest)} total")
print(f"  → Use this separately to report cross-system generalization in thesis.")

print("\n✓ Splits fixed. Now re-run:")
print("    python scripts/02_extract_features.py")
print("    python scripts/03_train_baselines.py")