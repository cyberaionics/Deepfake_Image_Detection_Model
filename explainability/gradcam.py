"""
Grad-CAM explainability for the Deepfake Detector.

Uses pytorch-grad-cam to generate class activation heatmaps showing which
facial regions most influence the model's deepfake prediction.
"""

import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.config import Config
from models.fusion_model import MultiBranchDetector
from datasets.video_dataset import get_val_transforms

logger = get_logger("gradcam")


class DeepfakeGradCAM:
    """
    Grad-CAM visualization for the spatial branch of the deepfake detector.

    Generates heatmaps highlighting which facial regions contribute most
    to the model's fake/real classification.
    """

    def __init__(self, model: MultiBranchDetector, device: str = "cpu"):
        """
        Args:
            model: Trained MultiBranchDetector.
            device: Torch device.
        """
        self.model = model.eval()
        self.device = device

        # Wrapper to extract spatial branch output through the full model
        # Grad-CAM needs to hook into the spatial backbone's last conv layer
        target_layer = model.get_spatial_gradcam_layer()
        self.cam = GradCAM(
            model=model,
            target_layers=[target_layer],
        )

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        original_image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            image_tensor: Preprocessed input tensor (1, 3, H, W).
            original_image: Original RGB image as numpy array (H, W, 3), values in [0, 1].

        Returns:
            Tuple of (heatmap, overlay, prediction_probability).
        """
        image_tensor = image_tensor.to(self.device)

        # Get prediction
        with torch.no_grad():
            logits = self.model(image_tensor)
            prob = torch.sigmoid(logits).item()

        # Generate Grad-CAM
        targets = [BinaryClassifierOutputTarget(1)]  # Target fake class
        grayscale_cam = self.cam(
            input_tensor=image_tensor,
            targets=targets,
        )

        # Get first (and only) image's CAM
        heatmap = grayscale_cam[0, :]

        # Overlay on original image
        overlay = show_cam_on_image(
            original_image.astype(np.float32),
            heatmap,
            use_rgb=True,
        )

        return heatmap, overlay, prob

    def visualize_and_save(
        self,
        image_path: str,
        save_path: str,
        cfg: Config,
    ):
        """
        Generate and save Grad-CAM visualization for an image file.

        Args:
            image_path: Path to input image.
            save_path: Path to save the visualization.
            cfg: Configuration for transforms.
        """
        # Load and preprocess image
        original = np.array(Image.open(image_path).convert("RGB"))
        original_normalized = original.astype(np.float32) / 255.0

        # Apply transforms
        transform = get_val_transforms(cfg)
        augmented = transform(image=original)
        image_tensor = augmented["image"].unsqueeze(0)

        # Generate heatmap
        heatmap, overlay, prob = self.generate_heatmap(image_tensor, original_normalized)

        # Determine label
        label = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob if prob > 0.5 else 1 - prob

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original)
        axes[0].set_title("Original Image", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", fontsize=13)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Overlay — {label} ({confidence:.1%})",
            fontsize=13,
            color="red" if label == "FAKE" else "green",
        )
        axes[2].axis("off")

        plt.suptitle(
            "Grad-CAM Explainability — Deepfake Detection",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Grad-CAM saved: {save_path} (pred={label}, conf={confidence:.4f})")

    def batch_visualize(
        self,
        image_dir: str,
        output_dir: str,
        cfg: Config,
        max_images: int = 20,
    ):
        """
        Generate Grad-CAM visualizations for multiple images.

        Args:
            image_dir: Directory containing images.
            output_dir: Directory to save visualizations.
            cfg: Configuration.
            max_images: Maximum number of images to process.
        """
        import glob

        os.makedirs(output_dir, exist_ok=True)
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        image_paths = image_paths[:max_images]

        for img_path in image_paths:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"gradcam_{filename}.png")
            self.visualize_and_save(img_path, save_path, cfg)
