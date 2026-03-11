"""
Spatial Feature Branch — EfficientNet-B4 Backbone.

Extracts spatial artifact features (texture inconsistencies, blending artifacts,
eye/mouth generation errors) from face images using a pretrained EfficientNet-B4.

Output embedding: 1792 dimensions.
"""

import torch
import torch.nn as nn
import timm


class SpatialBranch(nn.Module):
    """
    Spatial feature extraction using EfficientNet-B4.

    Uses the `timm` library to load a pretrained EfficientNet-B4 backbone,
    removes the classification head, and outputs a 1792-dim feature vector.
    """

    def __init__(
        self,
        pretrained: bool = True,
        embed_dim: int = 1792,
        freeze_early: bool = False,
    ):
        """
        Args:
            pretrained: Load ImageNet pretrained weights.
            embed_dim: Expected output embedding dimension (1792 for EfficientNet-B4).
            freeze_early: If True, freeze the first few blocks for fine-tuning.
        """
        super().__init__()

        # Load EfficientNet-B4 without classifier
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,        # Remove classifier head
            global_pool="avg",    # Global average pooling
        )

        self.embed_dim = embed_dim

        # Optionally freeze early layers
        if freeze_early:
            self._freeze_early_layers()

    def _freeze_early_layers(self):
        """Freeze the first 3 blocks of EfficientNet for fine-tuning stability."""
        for name, param in self.backbone.named_parameters():
            # Freeze stem and first 3 blocks
            if any(f"blocks.{i}" in name for i in range(3)) or "conv_stem" in name:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            Feature embedding of shape (B, 1792).
        """
        return self.backbone(x)

    def get_feature_layer(self):
        """Return the last convolutional layer for Grad-CAM."""
        # EfficientNet-B4's last conv block
        return self.backbone.conv_head
