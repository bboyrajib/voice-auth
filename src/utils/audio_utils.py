"""
src/utils/audio_utils.py
Utility functions for loading, trimming, resampling, and augmenting audio.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import tempfile
from pydub import AudioSegment


# ── Loading ────────────────────────────────────────────────────────────────────

def convert_to_wav(input_path):
    """
    Converts any supported audio file to WAV (PCM 16-bit).
    Returns path to converted WAV file.
    """
    suffix = Path(input_path).suffix.lower()

    if suffix == ".wav":
        return input_path  # already WAV

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(tmp_wav.name, format="wav")

    return tmp_wav.name


def load_audio_streamlit(path, target_sr=16000):
    """
    Always converts input to WAV first.
    Then loads safely using soundfile.
    """
    wav_path = convert_to_wav(path)

    audio, sr = sf.read(wav_path)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, mono=True)
        sr = target_sr

    return audio, sr

def load_audio(path: str, target_sr: int = 8000) -> tuple[np.ndarray, int]:
    """Load an audio file, resample to target_sr, convert to mono."""
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def save_audio(audio: np.ndarray, path: str, sr: int = 8000):
    """Save audio array to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


# ── Duration control ───────────────────────────────────────────────────────────

def trim_or_pad(audio: np.ndarray, sr: int, min_dur: float = 3.0, max_dur: float = 8.0) -> np.ndarray | None:
    """
    Trim audio to max_dur seconds. If shorter than min_dur, return None (discard).
    Pads short clips with silence if they're between min_dur and max_dur.
    """
    min_samples = int(min_dur * sr)
    max_samples = int(max_dur * sr)

    if len(audio) < min_samples:
        return None  # Too short — discard

    if len(audio) > max_samples:
        audio = audio[:max_samples]  # Trim to max

    # Pad to exactly max_dur with trailing silence
    if len(audio) < max_samples:
        pad = max_samples - len(audio)
        audio = np.concatenate([audio, np.zeros(pad)])

    return audio


# ── Noise augmentation ─────────────────────────────────────────────────────────

def add_white_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise at a specified SNR level (dB)."""
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    return audio + noise


def add_babble_noise(audio: np.ndarray, noise_clips: list[np.ndarray], snr_db: float) -> np.ndarray:
    """
    Add babble noise (mix of speech clips) at a specified SNR.
    noise_clips: list of noise waveforms (same sr as audio).
    """
    # Concatenate and tile noise to match signal length
    noise = np.concatenate(noise_clips)
    if len(noise) < len(audio):
        repeats = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:len(audio)]

    signal_power = np.mean(audio ** 2)
    noise_power_actual = np.mean(noise ** 2)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / (noise_power_actual + 1e-9))
    return audio + noise * scale


# ── Codec simulation ───────────────────────────────────────────────────────────

def simulate_g711(audio: np.ndarray, sr: int = 8000) -> np.ndarray:
    """
    Simulate G.711 mu-law codec compression.
    G.711 operates at 8kHz — audio must already be 8kHz.
    """
    assert sr == 8000, "G.711 requires 8kHz audio. Resample first."

    MU = 255.0
    # Normalize to [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)
    # Mu-law encode
    encoded = np.sign(audio) * np.log1p(MU * np.abs(audio)) / np.log1p(MU)
    # Quantize to 8-bit (256 levels)
    quantized = np.round(encoded * 127.5) / 127.5
    # Mu-law decode
    decoded = np.sign(quantized) * (1.0 / MU) * ((1 + MU) ** np.abs(quantized) - 1)
    return decoded.astype(np.float32)


# ── Batch processing ───────────────────────────────────────────────────────────

def get_audio_files(directory: str, extensions: tuple = (".wav", ".flac", ".mp3")) -> list[str]:
    """Recursively collect all audio files in a directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if f.lower().endswith(extensions):
                files.append(os.path.join(root, f))
    return sorted(files)


def get_duration(path: str, sr: int = 8000) -> float:
    """Return duration of an audio file in seconds without fully loading it."""
    info = sf.info(path)
    return info.duration


if __name__ == "__main__":
    # Quick sanity check
    print("audio_utils.py loaded successfully.")
    print("Testing G.711 codec simulation...")
    dummy = np.random.randn(8000).astype(np.float32) * 0.1
    decoded = simulate_g711(dummy, sr=8000)
    print(f"  Input shape: {dummy.shape}, Output shape: {decoded.shape}")
    print(f"  Max amplitude: {np.max(np.abs(decoded)):.4f}")
    print("OK.")
