"""
Test evaluation on FaceForensics++ C23 test split.

Loads the best trained model checkpoint, runs inference on the FF++ C23 test set,
computes all metrics, and generates visualization outputs.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from utils.logger import get_logger
from models.fusion_model import MultiBranchDetector
from datasets.video_dataset import DeepfakeDataset, get_val_transforms
from torch.utils.data import DataLoader
from evaluation.metrics import (
    compute_all_metrics,
    print_metrics_table,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics_table,
)

logger = get_logger("test_ffpp")


def load_model(cfg: Config, checkpoint_path: str) -> MultiBranchDetector:
    """
    Load model from checkpoint.

    Args:
        cfg: Configuration.
        checkpoint_path: Path to checkpoint file.

    Returns:
        Loaded model in eval mode.
    """
    model = MultiBranchDetector(
        spatial_embed_dim=cfg.spatial_embed_dim,
        frequency_embed_dim=cfg.frequency_embed_dim,
        srm_embed_dim=cfg.srm_embed_dim,
        hidden_dim=cfg.fusion_hidden_dim,
        dropout_rate=cfg.dropout_rate,
        pretrained=False,  # No need for pretrained; loading checkpoint
    )

    checkpoint = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(cfg.device)
    model.eval()

    logger.info(f"Loaded model from: {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    logger.info(f"Checkpoint best AUC: {checkpoint.get('best_auc', '?')}")

    return model


@torch.no_grad()
def evaluate_test_set(cfg: Config, checkpoint_path: str = None):
    """
    Full evaluation on FF++ C23 test split.

    Args:
        cfg: Configuration.
        checkpoint_path: Path to model checkpoint. Defaults to best_model.pt.
    """
    device = torch.device(cfg.device)

    # Default to best model checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.checkpoints_dir, "best_model.pt")

    if not os.path.isfile(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Load model
    model = load_model(cfg, checkpoint_path)

    # Create test dataset
    transform = get_val_transforms(cfg)
    test_dataset = DeepfakeDataset(
        root_dir=cfg.frames_dir,
        split="test",
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    logger.info(f"Test set size: {len(test_dataset)}")

    # Run inference
    all_probs = []
    all_labels = []

    for images, labels in tqdm(test_loader, desc="Evaluating FF++ C23 Test"):
        images = images.to(device, non_blocking=True)

        with autocast(device_type=str(device).split(":")[0], enabled=cfg.use_amp):
            logits = model(images)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().flatten().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = compute_all_metrics(all_labels, all_probs)

    # Print metrics
    print_metrics_table(metrics)

    # Generate output visualizations
    output_dir = os.path.join(cfg.outputs_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics table
    save_metrics_table(metrics, os.path.join(output_dir, "metrics.txt"))

    # Plot confusion matrix
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        os.path.join(output_dir, "confusion_matrix.png"),
    )

    # Plot ROC curve
    plot_roc_curve(
        all_labels, all_probs,
        os.path.join(output_dir, "roc_curve.png"),
    )

    logger.info(f"All evaluation outputs saved to: {output_dir}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate on FF++ C23 test set")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.data_root:
        cfg.frames_dir = args.data_root

    evaluate_test_set(cfg, checkpoint_path=args.checkpoint)
