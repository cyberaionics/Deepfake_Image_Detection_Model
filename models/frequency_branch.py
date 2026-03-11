"""
Frequency Feature Branch — FFT-based artifact detection.

Deepfake generation models introduce subtle artifacts in the frequency domain.
This branch converts images to frequency space via 2D FFT, computes the
log-scaled magnitude spectrum, and feeds it through a small CNN.

Output embedding: 256 dimensions.
"""

import torch
import torch.nn as nn


class FrequencyBranch(nn.Module):
    """
    Frequency domain feature extraction branch.

    Pipeline:
        1. Convert RGB input to grayscale
        2. Apply 2D FFT
        3. Compute magnitude spectrum with log scaling
        4. Pass through a small CNN → global average pool

    Output: 256-dim embedding.
    """

    def __init__(self, embed_dim: int = 256):
        """
        Args:
            embed_dim: Output embedding dimension.
        """
        super().__init__()

        self.embed_dim = embed_dim

        # Small CNN to process frequency magnitude maps
        self.cnn = nn.Sequential(
            # Input: (B, 1, H, W) — single-channel magnitude spectrum
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
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

    @staticmethod
    def compute_fft_magnitude(x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-scaled FFT magnitude spectrum.

        Args:
            x: Input tensor (B, 3, H, W) in RGB.

        Returns:
            Log-scaled magnitude spectrum (B, 1, H, W).
        """
        # Convert to grayscale: weighted average of RGB channels
        gray = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]
        gray = gray.unsqueeze(1)  # (B, 1, H, W)

        # Apply 2D FFT
        fft = torch.fft.fft2(gray)

        # Shift zero frequency to center
        fft_shifted = torch.fft.fftshift(fft)

        # Compute magnitude spectrum
        magnitude = torch.abs(fft_shifted)

        # Log scaling for better dynamic range
        magnitude_log = torch.log1p(magnitude)

        return magnitude_log

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency domain features.

        Args:
            x: Input tensor (B, 3, 224, 224).

        Returns:
            Feature embedding (B, 256).
        """
        # Compute FFT magnitude spectrum
        freq_map = self.compute_fft_magnitude(x)  # (B, 1, H, W)

        # Pass through CNN
        features = self.cnn(freq_map)  # (B, embed_dim, 1, 1)

        return features.flatten(1)  # (B, embed_dim)
