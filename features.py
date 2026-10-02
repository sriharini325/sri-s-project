"""
features.py
-----------
Turns a raw waveform into the numeric representations the models actually
learn from: MFCCs (for the classical ML baseline) and mel-spectrograms
(for the CNN).
"""

import librosa
import numpy as np


def extract_mfcc(signal, sr, n_mfcc=40):
    """
    Extracts MFCCs and summarizes them (mean + std per coefficient) into a
    fixed-length feature vector. Good input for SVM / Random Forest.

    Returns:
        np.ndarray of shape (n_mfcc * 2,)
    """
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    feature_vector = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feature_vector


def extract_melspectrogram(signal, sr, n_mels=128):
    """
    Extracts a log-mel-spectrogram: a 2D time-frequency "image" of the audio.
    Used as input to the CNN.

    Returns:
        np.ndarray of shape (n_mels, time_frames)
    """
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel
