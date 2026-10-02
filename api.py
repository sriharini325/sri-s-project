"""
api.py
------
Flask API alternative to the Streamlit app. Useful if you want to call
the detector programmatically (e.g. from another app or a test script).

Usage:
    python api.py

Then test with:
    curl -X POST -F "audio=@sample.wav" http://localhost:5000/predict
"""

import os
import torch
from flask import Flask, request, jsonify

from preprocessing import preprocess_audio
from features import extract_melspectrogram
from model import SpoofCNN

app = Flask(__name__)

model_path = "best_spoof_cnn.pt"
model = SpoofCNN()
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a .wav file via POST (multipart/form-data, field name "audio").
    Returns JSON: { "label": "spoof"/"bonafide", "confidence": float }
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided. Use field name 'audio'."}), 400

    audio_file = request.files["audio"]
    temp_path = "temp_api_audio.wav"
    audio_file.save(temp_path)

    try:
        signal, sr = preprocess_audio(temp_path)
        mel = extract_melspectrogram(signal, sr)
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            probs = torch.softmax(model(mel_tensor), dim=1).numpy()[0]

        label = "spoof" if probs[1] > probs[0] else "bonafide"
        confidence = float(max(probs) * 100)
        return jsonify({"label": label, "confidence": confidence})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
