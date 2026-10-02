"""
baseline_app.py
----------------
Streamlit web app for the SVM baseline pipeline (baseline_pipeline.py).
Uses only numpy/scipy/sklearn -- no librosa, no torch -- so it runs
anywhere Python runs.

Usage:
    streamlit run baseline_app.py
"""

import os
import joblib
import numpy as np
import streamlit as st

from baseline_pipeline import load_and_preprocess, extract_features

st.set_page_config(page_title="Synthetic Voice Detector (Baseline)", layout="centered")
st.title("Intelligent Detection of Synthetic Voices")
st.caption("Baseline model: SVM on spectral features")
st.write("Upload an audio clip to check whether it's a real human voice or an AI-generated/deepfake voice.")


@st.cache_resource
def load_trained_model():
    model_path = "baseline_model.joblib"
    if not os.path.exists(model_path):
        st.error(
            "No trained model found. Run these first in your terminal:\n\n"
            "1. python generate_sample_data.py\n"
            "2. python baseline_pipeline.py"
        )
        st.stop()
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["scaler"]


model, scaler = load_trained_model()

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("Analyzing voice..."):
        temp_path = "temp_upload.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        signal, sr = load_and_preprocess(temp_path)
        features = extract_features(signal, sr).reshape(1, -1)
        features_scaled = scaler.transform(features)

        probs = model.predict_proba(features_scaled)[0]  # [P(bonafide), P(spoof)]

        os.remove(temp_path)

    is_spoof = probs[1] > probs[0]
    confidence = float(max(probs) * 100)

    if is_spoof:
        st.error("SPOOF - Likely AI-Generated")
    else:
        st.success("BONAFIDE - Likely Human")

    st.write(f"Confidence: **{confidence:.1f}%**")
    st.progress(int(confidence))

    with st.expander("Raw probabilities"):
        st.write(f"P(bonafide): {probs[0]:.3f}")
        st.write(f"P(spoof): {probs[1]:.3f}")
