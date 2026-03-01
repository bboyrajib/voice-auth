# AI-Generated Speech Detection for Banking Voice Authentication

**Student:** Rajib Roy | **Student No.:** 24459 | **Stream:** AI | **IISc M.Tech Online**

---

## Project Goal
Build a binary classifier that distinguishes genuine human speech from AI-generated/cloned speech, evaluated specifically under banking telephony conditions (short phrases, codec compression, background noise).

---

## Project Structure

```
voiceauth_project/
├── data/
│   ├── raw/
│   │   ├── genuine/          # Downloaded real speech (LibriSpeech, VoxCeleb clips)
│   │   └── synthetic/        # AI-generated speech (Tacotron2, VITS, YourTTS)
│   ├── processed/
│   │   ├── train/            # Speaker-independent splits
│   │   ├── val/
│   │   └── test/
│   └── augmented/            # Noise-added, codec-compressed variants
├── src/
│   ├── features/             # Feature extraction modules
│   ├── models/               # ML and DL model definitions
│   ├── evaluation/           # Metrics, plots, EER computation
│   └── utils/                # Audio I/O, augmentation, helpers
├── notebooks/                # Jupyter notebooks (one per phase)
├── configs/                  # YAML config files
├── outputs/
│   ├── models/               # Saved model checkpoints
│   ├── results/              # CSVs of metric results
│   └── plots/                # ROC curves, confusion matrices
└── scripts/                  # Standalone runnable scripts
```

---

## Phases

| Phase | Term | Focus |
|-------|------|-------|
| 1 | Jan–Apr 2026 | Dataset + Feature Pipeline |
| 2 | May–Jul 2026 | ML Baselines + Deep Learning |
| 3 | Aug–Nov 2026 | Robustness + Interpretability + Thesis |

---

## Setup

```bash
# Create and activate environment
conda create -n voiceauth python=3.10
conda activate voiceauth

# Install dependencies
pip install -r requirements.txt
```

## Quick Start (after setup)

```bash
# Step 1: Download and prepare datasets
python scripts/01_download_data.py

# Step 2: Extract features
python scripts/02_extract_features.py

# Step 3: Train baseline models
python scripts/03_train_baselines.py

# Step 4: Train CNN model
python scripts/04_train_cnn.py

# Step 5: Evaluate all models
python scripts/05_evaluate.py
```

---

## Key Design Decisions

- **Speaker independence:** Train/val/test splits never share speakers — this is critical to avoid inflated metrics
- **Banking simulation:** All audio trimmed to 3–8s, downsampled to 8kHz, G.711 codec applied
- **EER as primary metric:** Equal Error Rate is the standard in anti-spoofing; accuracy alone is misleading
- **Reproducibility:** All random seeds fixed; configs stored in YAML; results logged to CSV
