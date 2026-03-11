"""
SRM Noise Residual Branch — Steganalysis Rich Model Filters.

Uses fixed (non-trainable) SRM convolution filters from digital image forensics
to capture noise residual patterns that reveal deepfake manipulation artifacts.

Output embedding: 128 dimensions.
"""

import torch
import torch.nn as nn
import numpy as np


class SRMBranch(nn.Module):
    """
    SRM noise residual feature extraction branch.

    Pipeline:
        1. Apply fixed SRM filters as non-trainable convolutions
        2. Pass noise residual maps through a small CNN
        3. Global average pool → 128-dim embedding

    The SRM filters detect high-pass noise patterns (edges, noise residuals)
    that differ between real and synthesized images.
    """

    def __init__(self, embed_dim: int = 128):
        """
        Args:
            embed_dim: Output embedding dimension.
        """
        super().__init__()

        self.embed_dim = embed_dim

        # Define SRM filters (fixed, non-trainable)
        self.srm_conv = self._build_srm_filters()

        # Small CNN to process noise residual maps
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),

            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1),
        )

    def _build_srm_filters(self) -> nn.Conv2d:
        """
        Build fixed SRM 5×5 convolution filters.

        Creates a Conv2d layer with non-trainable SRM filter weights.
        Three different SRM kernels are used (one per input RGB channel).

        Returns:
            Conv2d with fixed SRM filter weights.
        """
        # SRM Filter 1: Second-order edge detection
        srm_filter_1 = np.array([
            [0,  0,  0,  0,  0],
            [0, -1,  2, -1,  0],
            [0,  2, -4,  2,  0],
            [0, -1,  2, -1,  0],
            [0,  0,  0,  0,  0],
        ], dtype=np.float32)

        # SRM Filter 2: First-order horizontal
        srm_filter_2 = np.array([
            [0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0],
            [0,  1, -2,  1,  0],
            [0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0],
        ], dtype=np.float32)

        # SRM Filter 3: First-order diagonal
        srm_filter_3 = np.array([
            [0,  0,  0,  0,  0],
            [0, -1,  0,  1,  0],
            [0,  0,  0,  0,  0],
            [0,  1,  0, -1,  0],
            [0,  0,  0,  0,  0],
        ], dtype=np.float32)

        # Stack filters: (3, 1, 5, 5) → apply each filter to corresponding RGB channel
        # We'll use 3 input channels, 3 output channels (one SRM filter per channel)
        filters = np.stack([srm_filter_1, srm_filter_2, srm_filter_3])
        filters = filters[:, np.newaxis, :, :]  # (3, 1, 5, 5)

        # Build Conv2d: groups=3 means each filter applied to one input channel
        conv = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=3,
            bias=False,
        )

        # Set weights and freeze
        conv.weight = nn.Parameter(
            torch.from_numpy(filters).float(),
            requires_grad=False,
        )

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract SRM noise residual features.

        Args:
            x: Input tensor (B, 3, 224, 224).

        Returns:
            Feature embedding (B, 128).
        """
        # Apply fixed SRM filters → noise residual maps
        residual = self.srm_conv(x)  # (B, 3, H, W)

        # Pass through CNN
        features = self.cnn(residual)  # (B, embed_dim, 1, 1)

        return features.flatten(1)  # (B, embed_dim)
