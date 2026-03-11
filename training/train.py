"""
Training pipeline for the Multi-Branch Deepfake Detector.

Features:
    - BCEWithLogitsLoss
    - AdamW optimizer with cosine annealing
    - Mixed precision training (torch.cuda.amp)
    - Gradient accumulation
    - Checkpoint saving and resume
    - Reproducibility via seed setting
"""

import os
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from utils.logger import get_logger
from models.fusion_model import MultiBranchDetector
from datasets.video_dataset import create_dataloaders
from training.validation import validate

logger = get_logger("trainer")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_auc: float,
    save_path: str,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_auc": best_auc,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    checkpoint_path: str,
    device: str,
) -> tuple[int, float]:
    """
    Load training checkpoint for resume.

    Returns:
        Tuple of (start_epoch, best_auc).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint["epoch"]
    best_auc = checkpoint.get("best_auc", 0.0)

    logger.info(f"Resumed from checkpoint: epoch={epoch}, best_auc={best_auc:.4f}")
    return epoch + 1, best_auc


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
    epoch: int,
    cfg: Config,
) -> dict:
    """
    Train for one epoch.

    Returns:
        Dictionary with average loss and accuracy.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [Train]")

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixed precision forward
        device_type = str(device).split(":")[0]
        with autocast(device_type=device_type, enabled=cfg.use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
            loss = loss / cfg.gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Metrics
        total_loss += loss.item() * cfg.gradient_accumulation_steps
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{total_loss / (batch_idx + 1):.4f}",
            "acc": f"{correct / total:.4f}",
        })

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
    }


def train(cfg: Config, resume_path: str = None):
    """
    Full training loop.

    Args:
        cfg: Configuration object.
        resume_path: Path to checkpoint file for resuming training.
    """
    # Set seed for reproducibility
    cfg.set_seed()

    device = torch.device(cfg.device)
    device_str = str(device)
    logger.info(f"Training on device: {device}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    loaders = create_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Create model
    logger.info("Building multi-branch detector...")
    model = MultiBranchDetector(
        spatial_embed_dim=cfg.spatial_embed_dim,
        frequency_embed_dim=cfg.frequency_embed_dim,
        srm_embed_dim=cfg.srm_embed_dim,
        hidden_dim=cfg.fusion_hidden_dim,
        dropout_rate=cfg.dropout_rate,
        pretrained=True,
    ).to(device)

    # Log parameter counts
    param_counts = model.count_parameters()
    for branch, counts in param_counts.items():
        logger.info(f"  {branch}: {counts['trainable']:,} trainable / {counts['total']:,} total")

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.t_max)

    # Mixed precision scaler
    scaler = GradScaler(device_str, enabled=cfg.use_amp)

    # Resume from checkpoint
    start_epoch = 0
    best_auc = 0.0

    if resume_path and os.path.isfile(resume_path):
        start_epoch, best_auc = load_checkpoint(
            model, optimizer, scheduler, scaler, resume_path, cfg.device
        )

    # Training loop
    logger.info(f"Starting training: {cfg.num_epochs} epochs, batch_size={cfg.batch_size}")
    logger.info(f"{'═' * 70}")

    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device_str, epoch, cfg
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, cfg)

        # Step scheduler
        scheduler.step()

        # Logging
        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1}/{cfg.num_epochs} │ "
            f"Train Loss: {train_metrics['loss']:.4f} │ "
            f"Train Acc: {train_metrics['accuracy']:.4f} │ "
            f"Val Loss: {val_metrics['loss']:.4f} │ "
            f"Val Acc: {val_metrics['accuracy']:.4f} │ "
            f"Val AUC: {val_metrics['auc']:.4f} │ "
            f"LR: {scheduler.get_last_lr()[0]:.2e} │ "
            f"Time: {elapsed:.1f}s"
        )

        # Save checkpoint
        current_auc = val_metrics["auc"]

        # Save last checkpoint
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, best_auc,
            os.path.join(cfg.checkpoints_dir, "last_checkpoint.pt"),
        )

        # Save best checkpoint
        if current_auc > best_auc:
            best_auc = current_auc
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_auc,
                os.path.join(cfg.checkpoints_dir, "best_model.pt"),
            )
            logger.info(f"  ★ New best AUC: {best_auc:.4f}")

    logger.info(f"{'═' * 70}")
    logger.info(f"Training complete! Best validation AUC: {best_auc:.4f}")

    return model
