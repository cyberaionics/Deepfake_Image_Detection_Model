"""
Patch Importance explainability module.

Divides the image into patches, masks each patch, measures how the model's
prediction changes, and generates a patch importance heatmap.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.cuda.amp import autocast

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from utils.logger import get_logger
from models.fusion_model import MultiBranchDetector
from datasets.video_dataset import get_val_transforms

logger = get_logger("patch_importance")


class PatchImportanceAnalyzer:
    """
    Analyzes which image patches are most important for the model's prediction.

    Method:
        1. Get baseline prediction on the full image
        2. Divide image into a grid of patches
        3. For each patch, mask it (replace with gray) and re-run prediction
        4. Compute importance = |baseline_prob - masked_prob|
        5. Generate heatmap from importance values
    """

    def __init__(
        self,
        model: MultiBranchDetector,
        device: str = "cpu",
        patch_size: int = 32,
        mask_value: float = 0.5,
    ):
        """
        Args:
            model: Trained MultiBranchDetector.
            device: Torch device.
            patch_size: Size of each patch (pixels).
            mask_value: Value to fill masked patches with (before normalization).
        """
        self.model = model.eval()
        self.device = device
        self.patch_size = patch_size
        self.mask_value = mask_value

    @torch.no_grad()
    def _get_prediction(self, image_tensor: torch.Tensor) -> float:
        """Get prediction probability for an image tensor."""
        image_tensor = image_tensor.to(self.device)
        with autocast(enabled=True, device_type=str(self.device).split(":")[0]):
            logits = self.model(image_tensor)
        return torch.sigmoid(logits).item()

    def compute_importance(
        self,
        image_tensor: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute patch importance map.

        Args:
            image_tensor: Preprocessed image tensor (1, 3, H, W).

        Returns:
            Importance map (grid_h, grid_w) with values indicating
            how much each patch affects the prediction.
        """
        _, _, H, W = image_tensor.shape
        ps = self.patch_size

        grid_h = H // ps
        grid_w = W // ps

        # Baseline prediction
        baseline_prob = self._get_prediction(image_tensor)

        importance_map = np.zeros((grid_h, grid_w))

        for i in range(grid_h):
            for j in range(grid_w):
                # Create masked version
                masked = image_tensor.clone()
                y1 = i * ps
                y2 = min((i + 1) * ps, H)
                x1 = j * ps
                x2 = min((j + 1) * ps, W)

                # Replace patch with gray (normalized equivalent)
                masked[:, :, y1:y2, x1:x2] = 0.0  # Zero after normalization ≈ gray

                # Get prediction on masked image
                masked_prob = self._get_prediction(masked)

                # Importance = change in prediction
                importance_map[i, j] = abs(baseline_prob - masked_prob)

        return importance_map

    def visualize_and_save(
        self,
        image_path: str,
        save_path: str,
        cfg: Config,
    ):
        """
        Generate and save patch importance visualization.

        Args:
            image_path: Path to input image.
            save_path: Path to save visualization.
            cfg: Configuration.
        """
        # Load image
        original = np.array(Image.open(image_path).convert("RGB"))
        original_normalized = original.astype(np.float32) / 255.0

        # Preprocess
        transform = get_val_transforms(cfg)
        augmented = transform(image=original)
        image_tensor = augmented["image"].unsqueeze(0)

        # Compute importance
        importance = self.compute_importance(image_tensor)

        # Get prediction
        prob = self._get_prediction(image_tensor)
        label = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob if prob > 0.5 else 1 - prob

        # Upsample importance map to original image size
        importance_upsampled = np.kron(
            importance,
            np.ones((self.patch_size, self.patch_size)),
        )
        # Trim to match image size
        importance_upsampled = importance_upsampled[
            :original.shape[0], :original.shape[1]
        ]

        # Normalize importance for display
        if importance_upsampled.max() > 0:
            importance_norm = importance_upsampled / importance_upsampled.max()
        else:
            importance_norm = importance_upsampled

        # Create overlay
        heatmap_colored = plt.cm.jet(importance_norm)[:, :, :3]
        overlay = (0.6 * original_normalized + 0.4 * heatmap_colored).clip(0, 1)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original)
        axes[0].set_title("Original Image", fontsize=13)
        axes[0].axis("off")

        im = axes[1].imshow(importance_norm, cmap="jet", interpolation="nearest")
        axes[1].set_title("Patch Importance Map", fontsize=13)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Overlay — {label} ({confidence:.1%})",
            fontsize=13,
            color="red" if label == "FAKE" else "green",
        )
        axes[2].axis("off")

        plt.suptitle(
            f"Patch Importance Analysis (patch={self.patch_size}px)",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Patch importance saved: {save_path} (pred={label}, conf={confidence:.4f})")

    def batch_visualize(
        self,
        image_dir: str,
        output_dir: str,
        cfg: Config,
        max_images: int = 10,
    ):
        """
        Generate patch importance visualizations for multiple images.

        Args:
            image_dir: Directory containing images.
            output_dir: Directory to save visualizations.
            cfg: Configuration.
            max_images: Maximum images to process.
        """
        import glob

        os.makedirs(output_dir, exist_ok=True)
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        image_paths = image_paths[:max_images]

        for img_path in image_paths:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"patch_{filename}.png")
            self.visualize_and_save(img_path, save_path, cfg)
