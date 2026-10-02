"""
app.py
------
Streamlit web app for the Intelligent Synthetic Voice Detector.

Usage:
    streamlit run app.py
"""

import os
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st

from preprocessing import preprocess_audio
from features import extract_melspectrogram
from model import SpoofCNN

st.set_page_config(page_title="Synthetic Voice Detector", layout="centered")
st.title("Intelligent Detection of Synthetic Voices")
st.write("Upload an audio clip to check whether it's a real human voice or an AI-generated/deepfake voice.")


@st.cache_resource
def load_model():
    model = SpoofCNN()
    model_path = "best_spoof_cnn.pt"
    if not os.path.exists(model_path):
        st.error("No trained model found. Run `python train.py` first to create best_spoof_cnn.pt.")
        st.stop()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


model = load_model()

uploaded_file = st.file_uploader("Upload a .wav or .mp3 file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("Analyzing voice..."):
        temp_path = "temp_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        signal, sr = preprocess_audio(temp_path)
        mel = extract_melspectrogram(signal, sr)
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            logits = model(mel_tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        os.remove(temp_path)

    is_spoof = probs[1] > probs[0]
    label = "SPOOF - Likely AI-Generated" if is_spoof else "BONAFIDE - Likely Human"
    confidence = float(max(probs) * 100)

    if is_spoof:
        st.error(label)
    else:
        st.success(label)

    st.write(f"Confidence: **{confidence:.1f}%**")
    st.progress(int(confidence))

    st.write("Mel-spectrogram used for detection:")
    fig, ax = plt.subplots()
    librosa.display.specshow(mel, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    st.pyplot(fig)
