"""
dataset.py
----------
A PyTorch Dataset that reads .wav files from data/real and data/fake,
preprocesses them, and converts them into mel-spectrogram tensors.
"""

import os
import glob
import torch
from torch.utils.data import Dataset

from preprocessing import preprocess_audio
from features import extract_melspectrogram


class VoiceDataset(Dataset):
    """
    Expects this folder layout:
        data/real/*.wav  -> label 0 (bonafide)
        data/fake/*.wav  -> label 1 (spoof)
    """

    def __init__(self, real_dir, fake_dir):
        self.files = []
        self.labels = []

        real_files = glob.glob(os.path.join(real_dir, "*.wav"))
        fake_files = glob.glob(os.path.join(fake_dir, "*.wav"))

        if len(real_files) == 0 or len(fake_files) == 0:
            print(f"WARNING: found {len(real_files)} real files and "
                  f"{len(fake_files)} fake files. Make sure you've added "
                  f".wav files to data/real/ and data/fake/.")

        for f in real_files:
            self.files.append(f)
            self.labels.append(0)  # bonafide
        for f in fake_files:
            self.files.append(f)
            self.labels.append(1)  # spoof

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        signal, sr = preprocess_audio(self.files[idx])
        mel = extract_melspectrogram(signal, sr)
        mel_tensor = torch.tensor(mel).unsqueeze(0).float()  # add channel dim -> (1, n_mels, time)
        label = self.labels[idx]
        return mel_tensor, label
