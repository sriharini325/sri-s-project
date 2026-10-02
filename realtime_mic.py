"""
realtime_mic.py
---------------
Records a short clip from your microphone and runs it through the trained
model. This is the "real-time-ish" detection path (rolling short windows,
not literal sub-second streaming).

Usage:
    python realtime_mic.py
"""

import numpy as np
import sounddevice as sd
import torch

from features import extract_melspectrogram
from model import SpoofCNN


def record_and_predict(model, duration=4, sr=16000, device="cpu"):
    print(f"Recording {duration} seconds... speak now.")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    signal = audio.flatten()

    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))

    mel = extract_melspectrogram(signal, sr)
    mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        logits = model(mel_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    label = "SPOOF (AI-generated)" if probs[1] > probs[0] else "BONAFIDE (human)"
    confidence = max(probs) * 100
    return label, confidence


if __name__ == "__main__":
    model = SpoofCNN()
    model.load_state_dict(torch.load("best_spoof_cnn.pt", map_location="cpu"))

    label, confidence = record_and_predict(model)
    print(f"\nResult: {label}  (confidence: {confidence:.1f}%)")
