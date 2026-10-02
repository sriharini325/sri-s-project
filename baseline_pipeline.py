"""
baseline_pipeline.py
--------------------
A complete, self-contained detection pipeline (features -> train -> evaluate)
that uses ONLY numpy, scipy, and scikit-learn -- no librosa, no torch.

Why this file exists: librosa/torch installs can be slow or fail on some
setups. This baseline guarantees you have ONE fully working, verified path
from raw .wav files to a trained model and accuracy numbers, using
libraries that install almost anywhere. Use this to confirm your data is
good and your project runs end-to-end -- then move to the CNN
(model.py + train.py) for your main deep-learning results.

Pipeline stages:
  1. Data collection  -> reads data/real/*.wav and data/fake/*.wav
  2. Preprocessing    -> normalize, fix length
  3. Feature extraction -> hand-rolled spectral features (no librosa needed)
  4. Model selection   -> Support Vector Machine (SVM) classifier
  5. Model training    -> fit on 80% split
  6. Model testing     -> accuracy, precision, recall, F1, EER on 20% split

Usage:
    python baseline_pipeline.py
"""

import os
import glob
import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
import joblib


# ---------------------------------------------------------------------------
# 1 & 2. Data collection + preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(file_path, target_len_seconds=4.0):
    """
    Loads a .wav file, converts to float32 in [-1, 1], and pads/truncates
    to a fixed length so every clip produces a same-size feature vector.
    """
    sr, signal = wavfile.read(file_path)

    # Convert to float32 range [-1, 1] regardless of original bit depth
    if signal.dtype == np.int16:
        signal = signal.astype(np.float32) / 32768.0
    elif signal.dtype == np.int32:
        signal = signal.astype(np.float32) / 2147483648.0
    else:
        signal = signal.astype(np.float32)

    # If stereo, average to mono
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    target_len = int(sr * target_len_seconds)
    if len(signal) > target_len:
        signal = signal[:target_len]
    else:
        signal = np.pad(signal, (0, target_len - len(signal)))

    return signal, sr


# ---------------------------------------------------------------------------
# 3. Feature extraction (spectral features, no librosa required)
# ---------------------------------------------------------------------------

def extract_features(signal, sr, n_bands=40):
    """
    Extracts a fixed-length feature vector from a waveform using only
    numpy/scipy:
      - Log-energy in `n_bands` frequency bands (a simplified stand-in
        for mel-filterbank energies / MFCCs)
      - Spectral centroid, spectral flatness, zero-crossing rate
        (classic features known to help separate natural vs. synthetic speech)

    Returns a 1D numpy array (fixed length regardless of input length).
    """
    # Frame the signal and take the magnitude spectrum of the whole clip
    windowed = signal * np.hanning(len(signal))
    spectrum = np.abs(rfft(windowed))
    freqs = rfftfreq(len(signal), d=1.0 / sr)

    # Log-energy in evenly spaced frequency bands (0 - 8000 Hz)
    band_edges = np.linspace(0, sr / 2, n_bands + 1)
    band_energies = []
    for i in range(n_bands):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        energy = np.sum(spectrum[mask] ** 2) if mask.any() else 0.0
        band_energies.append(np.log1p(energy))
    band_energies = np.array(band_energies)

    # Spectral centroid: "brightness" of the sound
    centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-8)

    # Spectral flatness: how noise-like vs. tonal the spectrum is
    # (synthetic vocoders often have unnaturally flat or unnaturally
    # peaky spectra compared to real speech)
    geo_mean = np.exp(np.mean(np.log(spectrum + 1e-8)))
    arith_mean = np.mean(spectrum) + 1e-8
    flatness = geo_mean / arith_mean

    # Zero-crossing rate: how often the signal changes sign
    # (robotic/quantized audio often has an unusually regular ZCR)
    zcr = np.mean(np.abs(np.diff(np.sign(signal)))) / 2

    extra_features = np.array([centroid, flatness, zcr])
    return np.concatenate([band_energies, extra_features])


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def build_dataset(real_dir="data/real", fake_dir="data/fake"):
    real_files = sorted(glob.glob(os.path.join(real_dir, "*.wav")))
    fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.wav")))

    if len(real_files) == 0 or len(fake_files) == 0:
        raise RuntimeError(
            f"Found {len(real_files)} real and {len(fake_files)} fake files. "
            "Run `python generate_sample_data.py` first, or add your own "
            ".wav files to data/real/ and data/fake/."
        )

    X, y = [], []
    for f in real_files:
        signal, sr = load_and_preprocess(f)
        X.append(extract_features(signal, sr))
        y.append(0)  # bonafide
    for f in fake_files:
        signal, sr = load_and_preprocess(f)
        X.append(extract_features(signal, sr))
        y.append(1)  # spoof

    print(f"Loaded {len(real_files)} bonafide and {len(fake_files)} spoof clips.")
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# EER metric
# ---------------------------------------------------------------------------

def compute_eer(y_true, y_scores):
    """Equal Error Rate -- the standard anti-spoofing benchmark metric."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2 * 100


# ---------------------------------------------------------------------------
# 4, 5, 6. Model selection, training, testing
# ---------------------------------------------------------------------------

def run_pipeline():
    print("=== 1-2. Loading + preprocessing data ===")
    X, y = build_dataset()

    print("\n=== 3. Feature extraction complete ===")
    print(f"Feature vector length: {X.shape[1]}")

    print("\n=== 4-5. Model selection + training (SVM, RBF kernel) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling matters a lot for SVMs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    model.fit(X_train_scaled, y_train)

    print("\n=== 6. Testing / evaluation ===")
    y_pred = model.predict(X_test_scaled)
    y_scores = model.predict_proba(X_test_scaled)[:, 1]  # probability of "spoof"

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    eer = compute_eer(y_test, y_scores)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"EER:       {eer:.2f}%")
    print("Confusion matrix (rows=true, cols=predicted):")
    print("           bonafide  spoof")
    print(f"bonafide   {cm[0][0]:<9} {cm[0][1]}")
    print(f"spoof      {cm[1][0]:<9} {cm[1][1]}")

    # Save the trained model + scaler for reuse in the app
    joblib.dump({"model": model, "scaler": scaler}, "baseline_model.joblib")
    print("\nSaved trained model to baseline_model.joblib")

    return model, scaler


if __name__ == "__main__":
    run_pipeline()
