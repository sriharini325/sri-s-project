"""
preprocessing.py
----------------
Handles loading and cleaning raw audio files so every clip fed into the
model has a consistent sample rate, volume, and length.
"""

import librosa
import numpy as np


def preprocess_audio(file_path, target_sr=16000, duration=4.0):
    """
    Loads an audio file, resamples it, trims silence, normalizes volume,
    and pads/truncates it to a fixed length so every sample has the same shape.

    Args:
        file_path (str): path to a .wav/.mp3 file
        target_sr (int): sample rate to resample to (16kHz is standard for speech models)
        duration (float): fixed clip length in seconds

    Returns:
        signal (np.ndarray): 1D audio waveform, shape (target_sr * duration,)
        target_sr (int): the sample rate used
    """
    # 1. Load audio and resample to 16kHz (standard for speech models)
    signal, sr = librosa.load(file_path, sr=target_sr)

    # 2. Trim leading/trailing silence (simple Voice Activity Detection)
    signal, _ = librosa.effects.trim(signal, top_db=25)

    # 3. Normalize amplitude to range [-1, 1] so volume differences don't bias the model
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))

    # 4. Pad or truncate to a fixed number of samples
    target_len = int(target_sr * duration)
    if len(signal) > target_len:
        signal = signal[:target_len]  # truncate
    else:
        signal = np.pad(signal, (0, target_len - len(signal)))  # zero-pad

    return signal, target_sr
