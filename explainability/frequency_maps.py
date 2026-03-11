"""
Frequency domain visualization for deepfake detection explainability.

Visualizes FFT magnitude spectra and highlights abnormal high-frequency
artifacts present in fake images compared to real images.
"""

import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger

logger = get_logger("frequency_maps")


def compute_fft_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Compute log-scaled FFT magnitude spectrum of a grayscale image.

    Args:
        image: Input image (H, W, 3) in RGB or (H, W) grayscale.

    Returns:
        Log-scaled magnitude spectrum (H, W).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    gray = gray.astype(np.float32)

    # Apply 2D DFT
    dft = np.fft.fft2(gray)
    dft_shifted = np.fft.fftshift(dft)

    # Compute magnitude spectrum with log scaling
    magnitude = np.abs(dft_shifted)
    magnitude_log = np.log1p(magnitude)

    return magnitude_log


def compute_radial_average(spectrum: np.ndarray) -> np.ndarray:
    """
    Compute radial average of a 2D frequency spectrum.

    This helps compare frequency distributions between real and fake images.

    Args:
        spectrum: 2D magnitude spectrum.

    Returns:
        1D radial average profile.
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2

    # Create distance map from center
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = min(cy, cx)
    radial_avg = np.zeros(max_r)

    for ri in range(max_r):
        mask = r == ri
        if mask.any():
            radial_avg[ri] = spectrum[mask].mean()

    return radial_avg


def visualize_frequency_comparison(
    real_image_path: str,
    fake_image_path: str,
    save_path: str,
):
    """
    Generate side-by-side frequency spectrum comparison of real vs fake images.

    Args:
        real_image_path: Path to a real face image.
        fake_image_path: Path to a fake face image.
        save_path: Path to save the visualization.
    """
    real_img = np.array(Image.open(real_image_path).convert("RGB"))
    fake_img = np.array(Image.open(fake_image_path).convert("RGB"))

    real_spectrum = compute_fft_spectrum(real_img)
    fake_spectrum = compute_fft_spectrum(fake_img)

    # Compute difference to highlight artifacts
    # Normalize both spectra to the same scale
    max_val = max(real_spectrum.max(), fake_spectrum.max())
    real_norm = real_spectrum / max_val
    fake_norm = fake_spectrum / max_val
    diff = np.abs(fake_norm - real_norm)

    # Radial averages
    real_radial = compute_radial_average(real_spectrum)
    fake_radial = compute_radial_average(fake_spectrum)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Images and spectra
    axes[0, 0].imshow(real_img)
    axes[0, 0].set_title("Real Image", fontsize=13, color="green")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(fake_img)
    axes[0, 1].set_title("Fake Image", fontsize=13, color="red")
    axes[0, 1].axis("off")

    im = axes[0, 2].imshow(diff, cmap="hot")
    axes[0, 2].set_title("Frequency Difference", fontsize=13)
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 2: Frequency spectra and radial profiles
    axes[1, 0].imshow(real_spectrum, cmap="magma")
    axes[1, 0].set_title("Real FFT Spectrum", fontsize=13)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(fake_spectrum, cmap="magma")
    axes[1, 1].set_title("Fake FFT Spectrum", fontsize=13)
    axes[1, 1].axis("off")

    # Radial average comparison
    axes[1, 2].plot(real_radial, color="green", linewidth=2, label="Real", alpha=0.8)
    axes[1, 2].plot(fake_radial, color="red", linewidth=2, label="Fake", alpha=0.8)
    axes[1, 2].set_xlabel("Frequency (radius)", fontsize=12)
    axes[1, 2].set_ylabel("Log Magnitude", fontsize=12)
    axes[1, 2].set_title("Radial Frequency Profile", fontsize=13)
    axes[1, 2].legend(fontsize=11)
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(
        "Frequency Domain Analysis — Real vs Fake",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Frequency comparison saved: {save_path}")


def visualize_single_spectrum(image_path: str, save_path: str, title: str = ""):
    """
    Generate FFT magnitude spectrum visualization for a single image.

    Args:
        image_path: Path to input image.
        save_path: Path to save visualization.
        title: Optional title prefix.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    spectrum = compute_fft_spectrum(img)
    radial = compute_radial_average(spectrum)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title(f"{title} Image" if title else "Input Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(spectrum, cmap="magma")
    axes[1].set_title("FFT Magnitude Spectrum", fontsize=13)
    axes[1].axis("off")

    axes[2].plot(radial, color="#2563EB", linewidth=2)
    axes[2].set_xlabel("Frequency", fontsize=12)
    axes[2].set_ylabel("Log Magnitude", fontsize=12)
    axes[2].set_title("Radial Profile", fontsize=13)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f"Frequency Analysis{' — ' + title if title else ''}",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Frequency spectrum saved: {save_path}")
