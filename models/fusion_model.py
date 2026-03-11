"""
Multi-Branch Fusion Model for Deepfake Detection.

Fuses features from three branches:
    - Spatial (EfficientNet-B4):   1792 dims
    - Frequency (FFT + CNN):        256 dims
    - SRM (Noise Residual + CNN):   128 dims

Total fused feature vector: 2176 dims → FC classifier → binary output.
"""

import torch
import torch.nn as nn

from models.spatial_branch import SpatialBranch
from models.frequency_branch import FrequencyBranch
from models.srm_branch import SRMBranch


class MultiBranchDetector(nn.Module):
    """
    Multi-branch deepfake detector with feature-level fusion.

    Architecture:
        Input (B, 3, 224, 224)
            ├── SpatialBranch  → (B, 1792)
            ├── FrequencyBranch → (B, 256)
            └── SRMBranch      → (B, 128)
                                    ↓
            Concatenate → (B, 2176)
                                    ↓
            FC(2176→512) → ReLU → Dropout(0.3)
            FC(512→1)
                                    ↓
            Sigmoid → probability of being fake
    """

    def __init__(
        self,
        spatial_embed_dim: int = 1792,
        frequency_embed_dim: int = 256,
        srm_embed_dim: int = 128,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ):
        """
        Args:
            spatial_embed_dim: Spatial branch output dimension.
            frequency_embed_dim: Frequency branch output dimension.
            srm_embed_dim: SRM branch output dimension.
            hidden_dim: Hidden layer dimension in classifier.
            dropout_rate: Dropout rate.
            pretrained: Use pretrained EfficientNet-B4 weights.
        """
        super().__init__()

        # Feature extraction branches
        self.spatial_branch = SpatialBranch(
            pretrained=pretrained,
            embed_dim=spatial_embed_dim,
        )
        self.frequency_branch = FrequencyBranch(embed_dim=frequency_embed_dim)
        self.srm_branch = SRMBranch(embed_dim=srm_embed_dim)

        # Total fused dimension
        fused_dim = spatial_embed_dim + frequency_embed_dim + srm_embed_dim

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

        # Store dimensions for reference
        self.fused_dim = fused_dim
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass through all branches and classifier.

        Args:
            x: Input tensor (B, 3, 224, 224).
            return_features: If True, also return per-branch features.

        Returns:
            logits: Raw logits (B, 1) — pass through sigmoid for probability.
            features (optional): Dict with 'spatial', 'frequency', 'srm', 'fused' tensors.
        """
        # Extract features from each branch
        spatial_feat = self.spatial_branch(x)      # (B, 1792)
        frequency_feat = self.frequency_branch(x)  # (B, 256)
        srm_feat = self.srm_branch(x)              # (B, 128)

        # Concatenate
        fused = torch.cat([spatial_feat, frequency_feat, srm_feat], dim=1)  # (B, 2176)

        # Classify
        logits = self.classifier(fused)  # (B, 1)

        if return_features:
            features = {
                "spatial": spatial_feat,
                "frequency": frequency_feat,
                "srm": srm_feat,
                "fused": fused,
            }
            return logits, features

        return logits

    def get_spatial_gradcam_layer(self):
        """Return the target layer for Grad-CAM on the spatial branch."""
        return self.spatial_branch.get_feature_layer()

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.

        Args:
            x: Input tensor (B, 3, 224, 224).

        Returns:
            Probabilities (B, 1) in [0, 1].
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def count_parameters(self) -> dict[str, int]:
        """Count trainable and total parameters for each branch."""
        def _count(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return {"total": total, "trainable": trainable}

        return {
            "spatial": _count(self.spatial_branch),
            "frequency": _count(self.frequency_branch),
            "srm": _count(self.srm_branch),
            "classifier": _count(self.classifier),
            "full_model": _count(self),
        }
