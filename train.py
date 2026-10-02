"""
train.py
--------
Main training script. Run this after putting .wav files into
data/real/ and data/fake/.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import VoiceDataset
from model import SpoofCNN


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu"):
    """
    Standard supervised training loop with cross-entropy loss.
    Saves the best model (by validation accuracy) to best_spoof_cnn.pt.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- Validation ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds = model(x_val).argmax(dim=1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)
        val_acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.3f} | Val Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_spoof_cnn.pt")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.3f}")
    print("Best model saved to best_spoof_cnn.pt")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = VoiceDataset("data/real", "data/fake")

    if len(dataset) < 4:
        print("Not enough audio files found. Add .wav files to data/real/ and data/fake/ before training.")
    else:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)

        model = SpoofCNN()
        train_model(model, train_loader, val_loader, epochs=20, device=device)
