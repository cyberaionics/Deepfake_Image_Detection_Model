"""
Centralized configuration for the Deepfake Detection system.
All hyperparameters, paths, and settings are defined here.
"""

import os
import torch
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Master configuration for the deepfake detection pipeline."""

    # ── Paths ──────────────────────────────────────────────────────────
    project_root: str = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    data_root: str = ""
    frames_dir: str = ""
    checkpoints_dir: str = ""
    outputs_dir: str = ""

    # ── Dataset Sources (user must set these) ──────────────────────────
    ffpp_root: str = ""          # Path to FaceForensics++ root
    celebdf_root: str = ""       # Path to Celeb-DF v2 root
    train_csv: str = ""          # Path to CSV for training data
    val_csv: str = ""            # Path to CSV for validation data
    test_csv: str = ""           # Path to CSV for test data

    # ── Frame Extraction ───────────────────────────────────────────────
    frame_timestamps: list = field(default_factory=lambda: [0.5, 1.0, 1.5])
    frame_format: str = "png"

    # ── Face Preprocessing ─────────────────────────────────────────────
    face_detector: str = "mtcnn"  # "mtcnn" or "retinaface"
    face_size: int = 224
    face_margin: int = 20

    # ── Model Architecture ─────────────────────────────────────────────
    spatial_backbone: str = "efficientnet_b4"
    spatial_embed_dim: int = 1792
    frequency_embed_dim: int = 256
    srm_embed_dim: int = 128
    fusion_hidden_dim: int = 512
    dropout_rate: float = 0.3

    # ── Training ───────────────────────────────────────────────────────
    batch_size: int = 32
    num_epochs: int = 25
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 4
    pin_memory: bool = True
    gradient_accumulation_steps: int = 1

    # ── Mixed Precision ────────────────────────────────────────────────
    use_amp: bool = True

    # ── Scheduler ──────────────────────────────────────────────────────
    scheduler_type: str = "cosine"
    t_max: Optional[int] = None  # defaults to num_epochs

    # ── Augmentation ───────────────────────────────────────────────────
    aug_horizontal_flip_p: float = 0.5
    aug_color_jitter_p: float = 0.3
    aug_random_crop_p: float = 0.3
    aug_gaussian_blur_p: float = 0.2
    aug_jpeg_compression_p: float = 0.3
    aug_jpeg_quality_range: tuple = (30, 95)

    # ── Reproducibility ────────────────────────────────────────────────
    seed: int = 42

    # ── Device ─────────────────────────────────────────────────────────
    device: str = ""

    def __post_init__(self):
        # Resolve paths
        self.data_root = os.path.join(self.project_root, "datasets")
        self.frames_dir = os.path.join(self.data_root, "frames")
        self.checkpoints_dir = os.path.join(self.project_root, "checkpoints")
        self.outputs_dir = os.path.join(self.project_root, "outputs")

        # Auto-detect device
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Scheduler default
        if self.t_max is None:
            self.t_max = self.num_epochs

        # Create directories
        for d in [self.frames_dir, self.checkpoints_dir, self.outputs_dir]:
            os.makedirs(d, exist_ok=True)

    def set_seed(self):
        """Set random seeds for full reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_split_dir(self, split: str, label: str) -> str:
        """Get directory path for a specific split and label."""
        path = os.path.join(self.frames_dir, split, label)
        os.makedirs(path, exist_ok=True)
        return path

    def __repr__(self):
        lines = [f"{'─' * 50}", "  Deepfake Detector Configuration", f"{'─' * 50}"]
        for k, v in self.__dict__.items():
            lines.append(f"  {k:35s} : {v}")
        lines.append(f"{'─' * 50}")
        return "\n".join(lines)
