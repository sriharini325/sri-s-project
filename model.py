"""
model.py
--------
Defines the CNN architecture that classifies a mel-spectrogram as
bonafide (real human, label 0) or spoof (AI-generated, label 1).
"""

import torch
import torch.nn as nn


class SpoofCNN(nn.Module):
    """
    A compact CNN for classifying mel-spectrograms.
    Input shape: (batch, 1, n_mels, time_frames)
    Output: 2 logits -> [bonafide_score, spoof_score]
    """

    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((4, 4)),  # fixes output size regardless of input length
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),  # 2 classes: bonafide, spoof
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.classifier(x)
