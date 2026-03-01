"""
scripts/01b_expand_dataset.py  [overnight edition]
Expand synthetic speech dataset using three parallel sources.

Sources:
  A. Edge-TTS  — Microsoft neural TTS, 20 voices, 100 phrases = ~2000 clips (no GPU)
  B. VITS      — Coqui multi-speaker, 15 speakers, 100 phrases = ~1500 clips (espeak-ng ✓)
  C. YourTTS   — Zero-shot multi-lingual TTS, additional diversity (espeak-ng ✓)
  D. ASVspoof2019 — if already downloaded, processes automatically

Run overnight:
  python scripts/01b_expand_dataset.py --strategy all --max_edge 2000 --max_vits 1500

Quick test run first (recommended):
  python scripts/01b_expand_dataset.py --strategy edge_tts --max_edge 20
"""

import sys
import os
import json
import random
import argparse
import subprocess
import asyncio
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.utils.audio_utils import load_audio, save_audio, trim_or_pad

SYNTHETIC_DIR = ROOT / cfg["paths"]["data_raw_synthetic"]
GENUINE_DIR   = ROOT / cfg["paths"]["data_raw_genuine"]
PROC_DIR      = ROOT / cfg["paths"]["data_processed"]
TARGET_SR     = cfg["audio"]["target_sr"]
MIN_DUR       = cfg["audio"]["min_duration"]
MAX_DUR       = cfg["audio"]["max_duration"]
SEED          = cfg["project"]["seed"]

random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = SYNTHETIC_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Phrase bank: 100 banking utterances ───────────────────────────────────────

BANKING_PHRASES = [
    "Yes, I confirm this transaction.",
    "I authorize this payment.",
    "Please proceed with the transfer.",
    "Approved. Go ahead.",
    "Yes, that is correct.",
    "I confirm the withdrawal.",
    "Proceed with the transaction.",
    "Transfer approved.",
    "Yes, send it now.",
    "Confirm the payment please.",
    "I want to approve this transfer.",
    "Authorization granted.",
    "Yes, I consent to this transaction.",
    "Please complete the payment.",
    "I approve this.",
    "Go ahead with the transfer.",
    "Yes, that amount is correct.",
    "I authorize the withdrawal of funds.",
    "Confirmed. Please proceed.",
    "Yes, execute the transaction.",
    "My name is John Smith.",
    "I am calling to verify my account.",
    "My date of birth is January fifteen.",
    "The last four digits are four seven two one.",
    "I live at forty two Main Street.",
    "My account number ends in eight five.",
    "I want to access my savings account.",
    "Please verify my identity.",
    "I am the account holder.",
    "My pin is confirmed.",
    "The amount is five hundred dollars.",
    "I confirm the transfer of one thousand.",
    "Send two hundred and fifty dollars.",
    "The payment is for three hundred rupees.",
    "Transfer five thousand to my account.",
    "The amount is correct at seven fifty.",
    "I confirm one hundred and twenty dollars.",
    "Yes, the amount of sixty dollars is right.",
    "Transfer eight hundred and fifty dollars.",
    "The total is two thousand five hundred.",
    "Please block my card immediately.",
    "I want to activate my new card.",
    "Reset my internet banking password.",
    "Add a new beneficiary to my account.",
    "I want to increase my transaction limit.",
    "Please close this fixed deposit.",
    "Open a new savings account for me.",
    "Transfer funds to my joint account.",
    "Set up a standing instruction.",
    "I want to change my registered mobile number.",
    "I want to apply for a personal loan.",
    "Check my credit card outstanding balance.",
    "I need an increase in my credit limit.",
    "Pay the minimum due on my credit card.",
    "I want to foreclose my home loan.",
    "Show me my loan repayment schedule.",
    "I want to convert my purchases to EMI.",
    "Apply for an overdraft facility.",
    "Check the status of my loan application.",
    "I want to prepay my car loan.",
    "I did not make this transaction.",
    "Please freeze my account.",
    "I want to report a fraudulent charge.",
    "Someone has accessed my account without permission.",
    "Please reverse this unauthorized transaction.",
    "I want to file a dispute.",
    "Block all international transactions.",
    "Enable two factor authentication.",
    "Change my transaction password.",
    "I want to set a new security question.",
    "What is my account balance?",
    "Show me my last ten transactions.",
    "When is my next EMI due?",
    "What is the interest rate on my loan?",
    "How many reward points do I have?",
    "Please send my account statement.",
    "Update my email address.",
    "I want to nominate my spouse.",
    "Check if my cheque has been cleared.",
    "What is the minimum balance requirement?",
    "Yes I confirm that I want to transfer five hundred dollars.",
    "I authorize this payment of one thousand rupees for the electricity bill.",
    "Please proceed with the transfer of funds to my external account.",
    "I confirm my identity and authorize this high value transaction.",
    "Yes I am aware of the charges and I still want to proceed.",
    "I authorize the standing instruction for monthly rent payment.",
    "I want to approve the transaction and receive a confirmation SMS.",
    "I confirm that all the details entered are correct.",
    "Kindly debit my account and transfer the funds now.",
    "Yes this is my registered mobile number please proceed.",
    "I approve the beneficiary and confirm the transfer.",
    "Please do not cancel the transaction I want to continue.",
    "I have verified the details and I authorize this payment.",
    "Transfer the amount immediately to the linked account.",
    "I confirm the one time password is correct please proceed.",
    "Block the card and issue a new one to my address.",
    "I want to set the daily limit to ten thousand rupees.",
    "Debit my savings account for this transaction.",
    "I agree to the terms and conditions for this service.",
    "My voice is my password please verify me.",
]


# ── Edge-TTS: Microsoft neural TTS, 20 voices ─────────────────────────────────

async def _edge_synthesize(text, voice, out_path):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def generate_edge_tts(max_clips: int = 2000):
    try:
        import edge_tts
    except ImportError:
        print("[Edge-TTS] Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "edge-tts", "-q"])

    # 20 diverse neural voices across accents
    VOICES = [
        "en-US-AriaNeural",    "en-US-GuyNeural",
        "en-US-JennyNeural",   "en-US-DavisNeural",
        "en-US-AmberNeural",   "en-US-BrandonNeural",
        "en-GB-SoniaNeural",   "en-GB-RyanNeural",
        "en-GB-LibbyNeural",   "en-AU-NatashaNeural",
        "en-AU-WilliamNeural", "en-IN-NeerjaNeural",
        "en-IN-PrabhatNeural", "en-CA-ClaraNeural",
        "en-CA-LiamNeural",    "en-IE-EmilyNeural",
        "en-NZ-MollyNeural",   "en-SG-LunaNeural",
        "en-ZA-LeahNeural",    "en-US-MonicaNeural",
    ]

    raw_dir = OUT_DIR / "edge_tts_raw"
    raw_dir.mkdir(exist_ok=True)

    # Build all pairs, shuffle, cap at max_clips
    pairs = [(phrase, voice)
             for voice in VOICES
             for phrase in BANKING_PHRASES]
    random.shuffle(pairs)
    pairs = pairs[:max_clips]

    print(f"[Edge-TTS] Targeting {len(pairs)} clips across {len(VOICES)} voices...")
    manifest = []
    failed = 0

    for i, (phrase, voice) in enumerate(tqdm(pairs, desc="Edge-TTS")):
        out_path = OUT_DIR / f"edge_tts_{i:05d}.wav"
        if out_path.exists():          # resume-friendly
            manifest.append({
                "path": str(out_path), "label": 1,
                "label_str": "synthetic",
                "speaker_id": f"edge_{voice}",
                "source": "edge_tts", "phrase": phrase,
                "duration": MAX_DUR
            })
            continue

        raw_path = raw_dir / f"edge_{i:05d}.mp3"
        try:
            asyncio.run(_edge_synthesize(phrase, voice, raw_path))
            audio, sr = load_audio(str(raw_path), target_sr=TARGET_SR)
            audio = trim_or_pad(audio, TARGET_SR, MIN_DUR, MAX_DUR)
            if audio is None:
                raw_path.unlink(missing_ok=True)
                failed += 1
                continue
            save_audio(audio, str(out_path), sr=TARGET_SR)
            raw_path.unlink(missing_ok=True)
            manifest.append({
                "path": str(out_path), "label": 1,
                "label_str": "synthetic",
                "speaker_id": f"edge_{voice}",
                "source": "edge_tts", "phrase": phrase,
                "duration": len(audio) / TARGET_SR
            })
        except Exception as e:
            failed += 1
            if raw_path.exists():
                raw_path.unlink(missing_ok=True)

    print(f"[Edge-TTS] Done: {len(manifest)} clips generated, {failed} failed.")
    return manifest


# ── VITS: Coqui multi-speaker, 15 VCTK speakers ───────────────────────────────

def generate_vits(max_clips: int = 1500):
    try:
        from TTS.api import TTS as CoquiTTS
    except ImportError:
        print("[VITS] Installing Coqui TTS...")
        subprocess.run([sys.executable, "-m", "pip", "install", "TTS", "-q"])
        from TTS.api import TTS as CoquiTTS

    # 15 VCTK speakers — mix of male/female/accents
    SPEAKERS = [
        "p225", "p226", "p227", "p228", "p229",
        "p230", "p231", "p232", "p233", "p234",
        "p236", "p237", "p238", "p239", "p240",
    ]

    print(f"[VITS] Loading model (downloads ~500MB on first run)...")
    try:
        tts = CoquiTTS("tts_models/en/vctk/vits")
    except Exception as e:
        print(f"[VITS] Failed to load: {e}")
        return []

    clips_per_speaker = max_clips // len(SPEAKERS)
    phrases_to_use    = BANKING_PHRASES[:clips_per_speaker]

    manifest = []
    i = 0
    for spk in SPEAKERS:
        print(f"  VITS speaker {spk} ({clips_per_speaker} clips)...")
        for phrase in tqdm(phrases_to_use, desc=spk, leave=False):
            out_path = OUT_DIR / f"vits_{spk}_{i:05d}.wav"
            i += 1
            if out_path.exists():
                manifest.append({
                    "path": str(out_path), "label": 1,
                    "label_str": "synthetic",
                    "speaker_id": f"vits_{spk}",
                    "source": "vits", "phrase": phrase,
                    "duration": MAX_DUR
                })
                continue
            try:
                tts.tts_to_file(text=phrase, file_path=str(out_path), speaker=spk)
                audio, _ = load_audio(str(out_path), target_sr=TARGET_SR)
                audio = trim_or_pad(audio, TARGET_SR, MIN_DUR, MAX_DUR)
                if audio is None:
                    out_path.unlink(missing_ok=True)
                    continue
                save_audio(audio, str(out_path), sr=TARGET_SR)
                manifest.append({
                    "path": str(out_path), "label": 1,
                    "label_str": "synthetic",
                    "speaker_id": f"vits_{spk}",
                    "source": "vits", "phrase": phrase,
                    "duration": len(audio) / TARGET_SR
                })
            except Exception:
                if out_path.exists():
                    out_path.unlink(missing_ok=True)

    print(f"[VITS] Done: {len(manifest)} clips.")
    return manifest


# ── YourTTS: Zero-shot, additional TTS architecture diversity ──────────────────

def generate_yourtts(max_clips: int = 300):
    """
    YourTTS is a different architecture from VITS/Tacotron —
    adds TTS system diversity which helps model generalization.
    """
    try:
        from TTS.api import TTS as CoquiTTS
        tts = CoquiTTS("tts_models/multilingual/multi-dataset/your_tts")
    except Exception as e:
        print(f"[YourTTS] Skipping: {e}")
        return []

    print(f"[YourTTS] Generating {max_clips} clips...")
    manifest = []
    phrases  = (BANKING_PHRASES * 10)[:max_clips]

    for i, phrase in enumerate(tqdm(phrases, desc="YourTTS")):
        out_path = OUT_DIR / f"yourtts_{i:05d}.wav"
        if out_path.exists():
            manifest.append({
                "path": str(out_path), "label": 1,
                "label_str": "synthetic",
                "speaker_id": "yourtts_default",
                "source": "yourtts", "phrase": phrase,
                "duration": MAX_DUR
            })
            continue
        try:
            tts.tts_to_file(text=phrase, file_path=str(out_path),
                             language="en")
            audio, _ = load_audio(str(out_path), target_sr=TARGET_SR)
            audio = trim_or_pad(audio, TARGET_SR, MIN_DUR, MAX_DUR)
            if audio is None:
                out_path.unlink(missing_ok=True)
                continue
            save_audio(audio, str(out_path), sr=TARGET_SR)
            manifest.append({
                "path": str(out_path), "label": 1,
                "label_str": "synthetic",
                "speaker_id": "yourtts_default",
                "source": "yourtts", "phrase": phrase,
                "duration": len(audio) / TARGET_SR
            })
        except Exception:
            if out_path.exists():
                out_path.unlink(missing_ok=True)

    print(f"[YourTTS] Done: {len(manifest)} clips.")
    return manifest


# ── ASVspoof 2019 (if downloaded) ──────────────────────────────────────────────

def process_asvspooof_2019():
    asv_dir = ROOT / "data" / "raw" / "asvspooof2019"
    if not asv_dir.exists():
        print("[ASVspoof2019] Not found at data/raw/asvspooof2019/ — skipping.")
        print("  Register at: https://datashare.ed.ac.uk/handle/10283/3336")
        print("  Download LA.zip, extract, then re-run.")
        return []

    print("[ASVspoof2019] Processing...")
    manifest = []
    for subset in ["train", "dev"]:
        proto = (asv_dir / "LA" / "ASVspoof2019_LA_cm_protocols" /
                 f"ASVspoof2019.LA.cm.{subset}.trn.txt")
        audio_dir = asv_dir / "LA" / f"ASVspoof2019_LA_{subset}" / "flac"
        if not proto.exists():
            continue
        for line in tqdm(open(proto).readlines(), desc=f"ASV-{subset}"):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id, utt_id, _, system_id, label_str = parts[:5]
            label = 0 if label_str == "bonafide" else 1
            audio_path = audio_dir / f"{utt_id}.flac"
            if not audio_path.exists():
                continue
            try:
                out_path = OUT_DIR / f"asv19_{subset}_{utt_id}.wav"
                if not out_path.exists():
                    audio, _ = load_audio(str(audio_path), target_sr=TARGET_SR)
                    audio = trim_or_pad(audio, TARGET_SR, MIN_DUR, MAX_DUR)
                    if audio is None:
                        continue
                    save_audio(audio, str(out_path), sr=TARGET_SR)
                manifest.append({
                    "path": str(out_path), "label": label,
                    "label_str": label_str,
                    "speaker_id": f"asv19_{speaker_id}",
                    "source": f"asvspooof_{system_id}",
                    "duration": MAX_DUR
                })
            except Exception:
                continue
    g = sum(1 for m in manifest if m["label"]==0)
    s = sum(1 for m in manifest if m["label"]==1)
    print(f"[ASVspoof2019] {len(manifest)} clips: {g} genuine, {s} synthetic")
    return manifest


# ── Rebuild splits ─────────────────────────────────────────────────────────────

def rebuild_splits(new_manifests):
    genuine_path = GENUINE_DIR / "genuine_manifest.json"
    synth_path   = SYNTHETIC_DIR / "synthetic_manifest.json"

    with open(genuine_path) as f:
        genuine = json.load(f)

    existing_synth = []
    if synth_path.exists():
        with open(synth_path) as f:
            existing_synth = json.load(f)

    all_new = [item for m in new_manifests for item in m]

    # Merge + deduplicate by path
    seen, all_synthetic = set(), []
    for item in existing_synth + all_new:
        if item["path"] not in seen:
            seen.add(item["path"])
            all_synthetic.append(item)

    with open(synth_path, "w") as f:
        json.dump(all_synthetic, f, indent=2)

    g_total = len(genuine)
    s_total = len(all_synthetic)
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Genuine:   {g_total}")
    print(f"  Synthetic: {s_total}")
    print(f"  Ratio:     {g_total/max(s_total,1):.1f}:1")
    sources = {}
    for c in all_synthetic:
        sources[c['source']] = sources.get(c['source'], 0) + 1
    print(f"  Synthetic sources:")
    for src, cnt in sorted(sources.items()):
        print(f"    {src}: {cnt} clips")

    # Speaker-independent genuine split
    speakers = list(set(c["speaker_id"] for c in genuine))
    random.shuffle(speakers)
    n = len(speakers)
    train_spk = set(speakers[:int(n * 0.70)])
    val_spk   = set(speakers[int(n * 0.70):int(n * 0.85)])
    test_spk  = set(speakers[int(n * 0.85):])

    train_g = [c for c in genuine if c["speaker_id"] in train_spk]
    val_g   = [c for c in genuine if c["speaker_id"] in val_spk]
    test_g  = [c for c in genuine if c["speaker_id"] in test_spk]

    # Source-independent synthetic split: hold out one full source for test
    # This tests generalization to unseen TTS systems — important for your thesis
    sources_list = list(set(c["source"] for c in all_synthetic))
    print(f"\n  Splitting synthetic by source ({len(sources_list)} sources):")

    # Hold out edge_tts for test (largest, most diverse — good generalization test)
    # Use tacotron2 + vits for train, edge_tts for test
    if len(sources_list) >= 2:
        # Pick test source: prefer edge_tts if available (most different architecture)
        if "edge_tts" in sources_list:
            test_src = "edge_tts"
        else:
            test_src = sources_list[-1]

        non_test = [c for c in all_synthetic if c["source"] != test_src]
        test_s   = [c for c in all_synthetic if c["source"] == test_src]

        # Split non-test 85/15 into train/val
        random.shuffle(non_test)
        split_at = int(len(non_test) * 0.85)
        train_s = non_test[:split_at]
        val_s   = non_test[split_at:]

        print(f"    Train sources: {set(c['source'] for c in train_s)}")
        print(f"    Val sources:   {set(c['source'] for c in val_s)}")
        print(f"    Test source:   {test_src}  ← held-out for generalization test")
    else:
        random.shuffle(all_synthetic)
        ns = len(all_synthetic)
        train_s = all_synthetic[:int(ns*0.70)]
        val_s   = all_synthetic[int(ns*0.70):int(ns*0.85)]
        test_s  = all_synthetic[int(ns*0.85):]

    splits = {
        "train": train_g + train_s,
        "val":   val_g   + val_s,
        "test":  test_g  + test_s,
    }

    print(f"\n  Split breakdown:")
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        random.shuffle(items)
        with open(PROC_DIR / f"{name}_manifest.json", "w") as f:
            json.dump(items, f, indent=2)
        ng = sum(1 for c in items if c["label"]==0)
        ns = sum(1 for c in items if c["label"]==1)
        print(f"    {name:6s}: {len(items):5d} total | genuine={ng:4d} synthetic={ns:4d} | ratio={ng/max(ns,1):.1f}:1")

    print(f"\n✓ Splits rebuilt.")
    print(f"  Next steps:")
    print(f"    python scripts/02_extract_features.py")
    print(f"    python scripts/03_train_baselines.py")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all",
                         choices=["edge_tts", "vits", "yourtts", "asvspooof", "all"])
    parser.add_argument("--max_edge", type=int, default=2000,
                         help="Max Edge-TTS clips (20 voices × 100 phrases = 2000 max)")
    parser.add_argument("--max_vits", type=int, default=1500,
                         help="Max VITS clips (15 speakers × 100 phrases = 1500 max)")
    parser.add_argument("--max_yourtts", type=int, default=300)
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1b: Dataset Expansion  [overnight edition]")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Targets:  Edge-TTS={args.max_edge}  VITS={args.max_vits}  YourTTS={args.max_yourtts}")
    print()

    new_manifests = []

    if args.strategy in ("vits", "all"):
        m = generate_vits(max_clips=args.max_vits)
        if m:
            new_manifests.append(m)

    if args.strategy in ("yourtts", "all"):
        m = generate_yourtts(max_clips=args.max_yourtts)
        if m:
            new_manifests.append(m)

    if args.strategy in ("edge_tts", "all"):
        m = generate_edge_tts(max_clips=args.max_edge)
        if m:
            new_manifests.append(m)

    if args.strategy in ("asvspooof", "all"):
        m = process_asvspooof_2019()
        if m:
            new_manifests.append(m)

    if not new_manifests:
        print("\nNo new clips generated.")
        sys.exit(1)

    total = sum(len(m) for m in new_manifests)
    print(f"\nTotal new clips generated: {total}")
    rebuild_splits(new_manifests)
