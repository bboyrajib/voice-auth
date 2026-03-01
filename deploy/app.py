"""
app.py — AI Speech Detection Demo
Run: streamlit run app.py

Loads trained models and provides a web interface for classifying
audio files as genuine human speech or AI-generated/synthetic.
"""

import sys
import os
import tempfile
import numpy as np
import joblib
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from streamlit_mic_recorder import mic_recorder
from datetime import datetime


# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.extractor import (
    extract_all_handcrafted, extract_log_mel_spectrogram,
    normalize_spectrogram, get_feature_names
)
from src.utils.audio_utils import load_audio_streamlit, trim_or_pad
from src.models.cnn import SpeechCNN, SpeechCNNLSTM

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Speech Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px; padding: 20px; color: white;
    text-align: center; margin: 8px 0;
}
.genuine-card {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    border-radius: 12px; padding: 24px; color: white;
    text-align: center; font-size: 1.4em; font-weight: bold;
}
.synthetic-card {
    background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    border-radius: 12px; padding: 24px; color: white;
    text-align: center; font-size: 1.4em; font-weight: bold;
}
.uncertain-card {
    background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
    border-radius: 12px; padding: 24px; color: white;
    text-align: center; font-size: 1.4em; font-weight: bold;
}
.model-result { border-radius: 8px; padding: 12px; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    models = {}
    models_dir = ROOT / "outputs" / "models"

    # Classical models
    for name, key in [("SVM", "svm"), ("Random Forest", "random_forest"),
                       ("Gradient Boosting", "gradient_boosting")]:
        path = models_dir / f"{key}.joblib"
        if path.exists():
            artifact = joblib.load(path)
            models[name] = {"type": "classical", "model": artifact["model"],
                            "threshold": artifact["threshold"]}

    # Deep models
    for name, key, cls in [("CNN", "cnn", SpeechCNN),
                             ("CNN-LSTM", "cnn_lstm", SpeechCNNLSTM)]:
        path = models_dir / f"{key}_best.pt"
        if path.exists():
            m = cls()
            m.load_state_dict(torch.load(path, map_location="cpu"))
            m.eval()
            models[name] = {"type": "deep", "model": m}

    return models


@st.cache_data
def load_feature_names():
    p = ROOT / "data" / "processed" / "feature_names.json"
    if p.exists():
        import json
        with open(p) as f:
            return json.load(f)
    return get_feature_names(40)


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_one(audio, sr, model_info):
    """Run prediction on a single audio array. Returns (label, confidence, proba_synthetic)."""
    if model_info["type"] == "classical":
        hc = extract_all_handcrafted(audio, sr, n_mfcc=40)
        hc = hc.reshape(1, -1)
        proba = model_info["model"].predict_proba(hc)[0, 1]
        threshold = model_info["threshold"]
        label = "synthetic" if proba >= threshold else "genuine"
        conf = proba if label == "synthetic" else (1 - proba)
        return label, conf, proba

    else:  # deep
        spec = extract_log_mel_spectrogram(audio, sr, n_mels=128, n_fft=512, hop_length=128)
        spec = normalize_spectrogram(spec)[np.newaxis, np.newaxis, :, :]
        x = torch.tensor(spec, dtype=torch.float32)
        with torch.no_grad():
            logits = model_info["model"](x)
            proba_vec = torch.softmax(logits, dim=1)[0].numpy()
        proba = float(proba_vec[1])
        label = "synthetic" if proba >= 0.5 else "genuine"
        conf = max(proba_vec)
        return label, float(conf), proba


def ensemble_vote(results):
    """Weighted ensemble across all models."""
    weights = {"SVM": 1.5, "CNN-LSTM": 2.0, "CNN": 1.5,
               "Random Forest": 1.0, "Gradient Boosting": 1.0}
    total_w = 0
    weighted_proba = 0
    for name, res in results.items():
        w = weights.get(name, 1.0)
        weighted_proba += w * res["proba_synthetic"]
        total_w += w
    avg_proba = weighted_proba / total_w
    label = "synthetic" if avg_proba >= 0.5 else "genuine"
    conf = avg_proba if label == "synthetic" else (1 - avg_proba)
    return label, conf, avg_proba


# ── UI ────────────────────────────────────────────────────────────────────────
def main():

    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None

    if "audio_source" not in st.session_state:
        st.session_state.audio_source = None

    if "audio_format" not in st.session_state:
        st.session_state.audio_format = None

    if "audio_timestamp" not in st.session_state:
        st.session_state.audio_timestamp = None

    if "uploader_version" not in st.session_state:
        st.session_state.uploader_version = 0

        

    # Header
    st.title("🎙️ AI Speech Detector")
    st.markdown("**Banking Voice Authentication Security Tool** — Detects AI-generated and cloned speech")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        selected_models = st.multiselect(
            "Models to use",
            options=["SVM", "Random Forest", "Gradient Boosting", "CNN", "CNN-LSTM"],
            default=["SVM", "CNN-LSTM"],
            help="Select which models to run. Ensemble uses all selected models."
        )
        show_features = st.checkbox("Show feature breakdown", value=False)
        show_spectrogram = st.checkbox("Show spectrogram", value=True)
        target_sr = st.select_slider("Telephony mode (kHz)",
                                      options=[8, 16, 22], value=8,
                                      help="8 kHz = G.711 banking telephony, 16/22 kHz = higher quality")

        st.markdown("---")
        st.markdown("**Model Performance (Test EER)**")
        perf = {"SVM": 0.0083, "Random Forest": 0.0305, "Gradient Boosting": 0.0346,
                "CNN": 0.0081, "CNN-LSTM": 0.0035}
        for name, eer in perf.items():
            color = "🟢" if eer < 0.01 else "🟡" if eer < 0.04 else "🔴"
            st.markdown(f"{color} **{name}**: EER = {eer:.4f}")

        st.markdown("---")
        st.markdown("*M.Tech AI Project — IISc Bangalore*")
        st.markdown("*Rajib Roy, Student No. 24459*")

    # Load models
    with st.spinner("Loading models..."):
        all_models = load_all_models()

    if not all_models:
        st.error("❌ No trained models found in `outputs/models/`. Run the training scripts first.")
        return

    available = {k: v for k, v in all_models.items() if k in selected_models}
    if not available:
        st.warning("Select at least one model in the sidebar.")
        return

    # File upload
    col1, col2 = st.columns([2, 1])
    with col1:
        # st.subheader("📂 Upload Audio")

        st.subheader("🎤 Upload or Record Audio")

        tab1, tab2 = st.tabs(["📁 Upload File", "🎙 Record (Web)"])

        with tab1:
            uploaded = st.file_uploader(
                "Upload a WAV, MP3, or FLAC file",
                type=["wav", "mp3", "flac", "ogg", "m4a"],
                help="For best results use a short (3–8 second) utterance",
                key=f"audio_uploader_{st.session_state.uploader_version}"
            )
            if uploaded:
                st.session_state.audio_bytes = uploaded.read()
                st.session_state.audio_source = f"{uploaded.name}"
                st.session_state.audio_format = Path(uploaded.name).suffix.lower()
                st.session_state.audio_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with tab2:
            st.markdown("Click record and speak clearly (max 8 seconds).")

            recorded_audio = mic_recorder(
                start_prompt="🎤 Start Recording",
                stop_prompt="⏹ Stop Recording",
                just_once=True,
                use_container_width=True,
                format="wav"
            )

            # Save permanently
            if recorded_audio:
                if isinstance(recorded_audio, dict):
                    st.session_state.audio_bytes = recorded_audio["bytes"]
                else:
                    st.session_state.audio_bytes = recorded_audio

                st.session_state.audio_source = "Live Microphone Recording"
                st.session_state.audio_format = ".wav"
                st.session_state.audio_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 🔥 Reset uploader safely
                st.session_state.uploader_version += 1

                st.rerun()

        # Retrieve from session
        audio_bytes = st.session_state.audio_bytes
        audio_source = st.session_state.audio_source
        audio_format = st.session_state.audio_format
        audio_ts = st.session_state.audio_timestamp
        

    # if recorded_audio:
    #     st.audio(recorded_audio, format="audio/wav")

    with col2:
        st.subheader("ℹ️ About")
        st.info("""
**What this detects:**
- AI-generated TTS speech
- Voice-cloned audio
- Synthetic voice impersonation

**How it works:**
Extracts acoustic features (MFCCs, spectral shape, phase) and applies trained ML/DL classifiers.
        """)

    if audio_bytes is None:
        st.markdown("---")
        st.markdown("### 🎯 How to use")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1. Upload audio**\nDrop a WAV/MP3 file containing a voice sample.")
        with c2:
            st.markdown("**2. Select models**\nChoose which models to run in the sidebar.")
        with c3:
            st.markdown("**3. View results**\nSee per-model confidence scores and ensemble verdict.")
        return

    # Process audio
    with tempfile.NamedTemporaryFile(suffix=audio_format,delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with st.spinner("Processing audio..."):
            audio, sr = load_audio_streamlit(tmp_path, target_sr=target_sr * 1000)
            # Compute real duration first
            original_duration = len(audio) / sr
            audio_padded = trim_or_pad(audio, sr, min_dur=1.0, max_dur=8.0)
            if audio_padded is None:
                audio_padded = audio
            duration = len(audio_padded) / sr

        st.markdown("---")

        # Audio info + waveform
        info_col, wave_col = st.columns([1, 2])
        with info_col:
            st.markdown("**Audio Information**")
            st.markdown(f"- Source: `{audio_source}`")
            st.markdown(f"- Duration: {original_duration:.2f}s")
            st.markdown(f"- Sample rate: {sr:,} Hz ({target_sr} kHz telephony)")
            st.markdown(f"- Uploaded at: {audio_ts}")
            st.audio(tmp_path, format="audio/wav")

        with wave_col:
            times = np.linspace(0, duration, len(audio_padded))
            fig_wave = go.Figure()
            fig_wave.add_trace(go.Scatter(x=times, y=audio_padded, mode='lines',
                                           line=dict(color='#667eea', width=0.8), name='Waveform'))
            fig_wave.update_layout(title="Waveform", xaxis_title="Time (s)",
                                    yaxis_title="Amplitude", height=180,
                                    margin=dict(l=40, r=20, t=40, b=40),
                                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_wave, use_container_width=True)

        # Spectrogram
        if show_spectrogram:
            spec = extract_log_mel_spectrogram(audio_padded, sr, n_mels=128, n_fft=512, hop_length=128)
            fig_spec = px.imshow(spec, aspect="auto", color_continuous_scale="magma",
                                  labels={"x": "Time Frame", "y": "Mel Bin", "color": "Log Energy"},
                                  title="Log-Mel Spectrogram (input to CNN models)")
            fig_spec.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_spec, use_container_width=True)

        # Run all models
        st.markdown("---")
        st.subheader("🔍 Classification Results")

        results = {}
        progress = st.progress(0)
        for i, (name, model_info) in enumerate(available.items()):
            with st.spinner(f"Running {name}..."):
                label, conf, proba_synthetic = predict_one(audio_padded, sr, model_info)
                results[name] = {"label": label, "confidence": conf,
                                  "proba_synthetic": proba_synthetic}
            progress.progress((i + 1) / len(available))
        progress.empty()

        # Ensemble verdict
        ens_label, ens_conf, ens_proba = ensemble_vote(results)

        # Big verdict card
        verdict_pct = ens_proba * 100
        if verdict_pct > 70:
            card_class = "synthetic-card"
            verdict_icon = "🤖"
            verdict_text = "AI-GENERATED SPEECH DETECTED"
        elif verdict_pct < 30:
            card_class = "genuine-card"
            verdict_icon = "✅"
            verdict_text = "GENUINE HUMAN SPEECH"
        else:
            card_class = "uncertain-card"
            verdict_icon = "⚠️"
            verdict_text = "UNCERTAIN — MANUAL REVIEW RECOMMENDED"

        st.markdown(f"""
        <div class="{card_class}">
            {verdict_icon} &nbsp; {verdict_text}<br>
            <span style="font-size:0.7em; opacity:0.9">
                Synthetic probability: {verdict_pct:.1f}% &nbsp;|&nbsp;
                Ensemble confidence: {ens_conf*100:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Per-model results
        st.markdown("**Per-Model Results**")
        cols = st.columns(len(results))
        for col, (name, res) in zip(cols, results.items()):
            with col:
                pct = res["proba_synthetic"] * 100
                color = "#eb3349" if res["label"] == "synthetic" else "#11998e"
                icon = "🤖" if res["label"] == "synthetic" else "✅"
                st.markdown(f"""
                <div style="background:{color}22; border:2px solid {color};
                     border-radius:10px; padding:14px; text-align:center;">
                    <div style="font-size:1.1em; font-weight:bold; color:{color}">
                        {icon} {name}
                    </div>
                    <div style="font-size:0.85em; color:#333; margin-top:6px">
                        <b>{'Synthetic' if res['label']=='synthetic' else 'Genuine'}</b><br>
                        Synthetic prob: {pct:.1f}%<br>
                        Confidence: {res['confidence']*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Confidence gauge chart
        st.markdown("")
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ens_proba * 100,
            title={"text": "Synthetic Probability (%) — Ensemble", "font": {"size": 16}},
            delta={"reference": 50, "increasing": {"color": "#eb3349"}, "decreasing": {"color": "#11998e"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#eb3349" if ens_proba > 0.5 else "#11998e"},
                "steps": [
                    {"range": [0, 30], "color": "#d5f5e3"},
                    {"range": [30, 70], "color": "#fef9e7"},
                    {"range": [70, 100], "color": "#fadbd8"},
                ],
                "threshold": {"line": {"color": "gray", "width": 3},
                              "thickness": 0.75, "value": 50},
            }
        ))
        gauge_fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))

        bar_fig = go.Figure()
        colours = {"SVM": "#2196F3", "Random Forest": "#4CAF50",
                   "Gradient Boosting": "#FF9800", "CNN": "#9C27B0", "CNN-LSTM": "#F44336"}
        bar_fig.add_trace(go.Bar(
            x=list(results.keys()),
            y=[r["proba_synthetic"] * 100 for r in results.values()],
            marker_color=[colours.get(n, "#999") for n in results.keys()],
            text=[f"{r['proba_synthetic']*100:.1f}%" for r in results.values()],
            textposition="outside"
        ))
        bar_fig.add_hline(y=50, line_dash="dash", line_color="gray",
                           annotation_text="Decision boundary (50%)")
        bar_fig.update_layout(title="Synthetic Probability by Model (%)",
                               yaxis=dict(range=[0, 110]), height=280,
                               margin=dict(l=20, r=20, t=50, b=40),
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        gcol, bcol = st.columns(2)
        with gcol:
            st.plotly_chart(gauge_fig, use_container_width=True)
        with bcol:
            st.plotly_chart(bar_fig, use_container_width=True)

        # Feature breakdown
        if show_features:
            st.markdown("---")
            st.subheader("🔬 Feature Analysis")
            feature_names = load_feature_names()
            hc = extract_all_handcrafted(audio_padded, sr, n_mfcc=40)

            # Top discriminative features (from RF importance)
            top_features = [
                "delta_mfcc_6_mean", "delta_mfcc_5_mean", "mfcc_12_std",
                "delta_mfcc_8_mean", "delta_mfcc_9_mean", "phase_diff_mean",
                "centroid_mean", "rolloff_mean", "bandwidth_mean", "zcr_mean"
            ]
            indices = []
            vals = []
            labels_feat = []
            for feat in top_features:
                if feat in feature_names:
                    idx = feature_names.index(feat)
                    indices.append(idx)
                    vals.append(float(hc[idx]))
                    labels_feat.append(feat)

            if vals:
                fig_feat = go.Figure(go.Bar(
                    x=labels_feat, y=vals,
                    marker_color=["#2196F3" if "mfcc" in f else "#FF9800"
                                   if any(k in f for k in ["centroid","rolloff","bandwidth","zcr"])
                                   else "#4CAF50" for f in labels_feat]
                ))
                fig_feat.update_layout(
                    title="Key Feature Values (top discriminative features)",
                    xaxis_tickangle=-35, height=300,
                    margin=dict(l=20, r=20, t=50, b=80),
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_feat, use_container_width=True)
                st.caption("Blue = MFCC features | Orange = Spectral features | Green = Phase/Pitch features")

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
