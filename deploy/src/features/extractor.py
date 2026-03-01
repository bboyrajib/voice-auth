"""
src/features/extractor.py
Extracts handcrafted acoustic features and log-Mel spectrograms from audio.
All features are designed to capture properties that differ between genuine
and AI-generated speech — see proposal Section 5, Stage 2.
"""

import numpy as np
import librosa


# ── Handcrafted feature extraction ────────────────────────────────────────────

def extract_mfcc_features(audio: np.ndarray, sr: int, n_mfcc: int = 40,
                            include_delta: bool = True,
                            include_delta_delta: bool = True) -> np.ndarray:
    """
    Extract MFCC features and their deltas.
    Returns a 1D feature vector (mean + std over time for each coefficient).
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    features = [mfcc.mean(axis=1), mfcc.std(axis=1)]

    if include_delta:
        delta = librosa.feature.delta(mfcc)
        features += [delta.mean(axis=1), delta.std(axis=1)]

    if include_delta_delta:
        delta2 = librosa.feature.delta(mfcc, order=2)
        features += [delta2.mean(axis=1), delta2.std(axis=1)]

    return np.concatenate(features)


def extract_spectral_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract spectral shape features:
      - Spectral centroid, bandwidth, rolloff (mean + std)
      - Zero-crossing rate (mean + std)
      - Harmonic-to-noise ratio (mean)
    """
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)

    # HNR via harmonic decomposition
    harmonic, _ = librosa.effects.hpss(audio)
    noise_energy = np.mean((audio - harmonic) ** 2) + 1e-9
    harmonic_energy = np.mean(harmonic ** 2)
    hnr = 10 * np.log10(harmonic_energy / noise_energy)

    features = np.array([
        centroid.mean(), centroid.std(),
        bandwidth.mean(), bandwidth.std(),
        rolloff.mean(), rolloff.std(),
        zcr.mean(), zcr.std(),
        hnr
    ])
    return features


def extract_pitch_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract pitch (F0) statistics.
    Uses librosa's pyin for robust pitch tracking.
    Returns: mean, std, min, max of voiced F0, and voiced frame ratio.
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'), sr=sr
    )
    f0_voiced = f0[voiced_flag]

    if len(f0_voiced) == 0:
        return np.zeros(5)

    voiced_ratio = voiced_flag.sum() / len(voiced_flag)
    return np.array([
        f0_voiced.mean(),
        f0_voiced.std(),
        f0_voiced.min(),
        f0_voiced.max(),
        voiced_ratio
    ])


def extract_phase_features(audio: np.ndarray, sr: int, n_fft: int = 512,
                             hop_length: int = 128) -> np.ndarray:
    """
    Extract phase-based features.
    AI-generated speech often has smoother, more regular phase compared to natural speech.
    Returns: mean and std of instantaneous phase differences (group delay proxy).
    """
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    phase = np.angle(stft)
    # Instantaneous frequency: phase difference across time
    phase_diff = np.diff(phase, axis=1)
    # Unwrap to reduce discontinuities
    phase_diff_unwrapped = np.unwrap(phase_diff, axis=1)

    return np.array([
        phase_diff_unwrapped.mean(),
        phase_diff_unwrapped.std(),
        np.abs(phase_diff_unwrapped).mean()
    ])


def extract_all_handcrafted(audio: np.ndarray, sr: int,
                              n_mfcc: int = 40) -> np.ndarray:
    """
    Master function: extract all handcrafted features and concatenate.
    Output is a single 1D feature vector suitable for classical ML models.
    """
    mfcc_feats = extract_mfcc_features(audio, sr, n_mfcc=n_mfcc)
    spectral_feats = extract_spectral_features(audio, sr)
    pitch_feats = extract_pitch_features(audio, sr)
    phase_feats = extract_phase_features(audio, sr)

    return np.concatenate([mfcc_feats, spectral_feats, pitch_feats, phase_feats])


def get_feature_names(n_mfcc: int = 40) -> list[str]:
    """Return a list of feature names matching extract_all_handcrafted output."""
    names = []
    for stat in ["mean", "std"]:
        names += [f"mfcc_{i}_{stat}" for i in range(n_mfcc)]
    for stat in ["mean", "std"]:
        names += [f"delta_mfcc_{i}_{stat}" for i in range(n_mfcc)]
    for stat in ["mean", "std"]:
        names += [f"delta2_mfcc_{i}_{stat}" for i in range(n_mfcc)]
    names += ["centroid_mean", "centroid_std",
              "bandwidth_mean", "bandwidth_std",
              "rolloff_mean", "rolloff_std",
              "zcr_mean", "zcr_std",
              "hnr"]
    names += ["f0_mean", "f0_std", "f0_min", "f0_max", "voiced_ratio"]
    names += ["phase_diff_mean", "phase_diff_std", "phase_diff_abs_mean"]
    return names


# ── Spectrogram extraction ─────────────────────────────────────────────────────

def extract_log_mel_spectrogram(audio: np.ndarray, sr: int,
                                  n_mels: int = 128,
                                  n_fft: int = 512,
                                  hop_length: int = 128,
                                  target_frames: int = 128) -> np.ndarray:
    """
    Extract a log-Mel spectrogram, fixed to (n_mels, target_frames).
    Used as input to CNN models.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Crop or pad to fixed width
    if log_mel.shape[1] > target_frames:
        log_mel = log_mel[:, :target_frames]
    else:
        pad = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant", constant_values=-80.0)

    return log_mel  # Shape: (n_mels, target_frames)


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Normalize spectrogram to zero mean, unit variance."""
    mean = spec.mean()
    std = spec.std() + 1e-9
    return (spec - mean) / std


if __name__ == "__main__":
    print("extractor.py — running self-test...")
    sr = 8000
    duration = 5.0
    dummy_audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.05

    feats = extract_all_handcrafted(dummy_audio, sr)
    names = get_feature_names()
    print(f"  Handcrafted feature vector length: {len(feats)}")
    print(f"  Feature names count: {len(names)}")
    assert len(feats) == len(names), "Mismatch between features and names!"

    spec = extract_log_mel_spectrogram(dummy_audio, sr)
    print(f"  Log-Mel spectrogram shape: {spec.shape}")
    assert spec.shape == (128, 128), "Unexpected spectrogram shape!"

    print("All checks passed.")
