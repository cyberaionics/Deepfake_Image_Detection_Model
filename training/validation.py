"""
Validation loop for the Deepfake Detector.

Runs model evaluation on the validation set with metric computation.
"""

import torch
import torch.nn as nn
from torch.amp import autocast
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from utils.logger import get_logger

logger = get_logger("validator")


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: str | torch.device,
    epoch: int,
    cfg: Config,
) -> dict:
    """
    Run validation and compute metrics.

    Args:
        model: Trained model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Device string or torch.device.
        epoch: Current epoch number.
        cfg: Configuration object.

    Returns:
        Dictionary with loss, accuracy, and AUC-ROC.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [Val]")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=str(device).split(":")[0], enabled=cfg.use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_probs.extend(probs.cpu().numpy().flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

        pbar.set_postfix({
            "loss": f"{total_loss / (len(all_probs) // cfg.batch_size + 1):.4f}",
            "acc": f"{correct / total:.4f}",
        })

    # Compute AUC-ROC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
        logger.warning("AUC-ROC could not be computed (single class in batch)")

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
        "auc": auc,
        "probs": all_probs,
        "labels": all_labels,
    }
