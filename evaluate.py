"""
evaluate.py
-----------
Metrics used to judge the trained model:
- Accuracy / Precision / Recall / F1 (general classification quality)
- Confusion matrix
- EER (Equal Error Rate) - the standard metric in anti-spoofing research

Run this after training to get a full report on your validation/test set.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)

from dataset import VoiceDataset
from model import SpoofCNN


def compute_eer(y_true, y_scores):
    """
    Computes the Equal Error Rate (EER) - the point where the false-acceptance
    rate equals the false-rejection rate. Lower EER = better.

    y_scores should be the model's predicted probability of the 'spoof' class.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    return eer * 100  # as a percentage


def evaluate_model(model_path="best_spoof_cnn.pt", real_dir="data/real", fake_dir="data/fake"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SpoofCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    dataset = VoiceDataset(real_dir, fake_dir)
    loader = DataLoader(dataset, batch_size=16)

    all_labels = []
    all_preds = []
    all_spoof_scores = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_labels.extend(y_batch.numpy())
            all_preds.extend(preds)
            all_spoof_scores.extend(probs[:, 1])  # probability of "spoof" class

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    eer = compute_eer(all_labels, all_spoof_scores)

    print("===== Evaluation Report =====")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"EER:       {eer:.2f}%")
    print("Confusion Matrix (rows=true, cols=predicted):")
    print("           bonafide  spoof")
    print(f"bonafide   {cm[0][0]:<9} {cm[0][1]}")
    print(f"spoof      {cm[1][0]:<9} {cm[1][1]}")


if __name__ == "__main__":
    evaluate_model()
