"""
Evaluation metrics for the Deepfake Detector.

Computes:
    - Accuracy (%)
    - AUC-ROC
    - F1-score (macro)
    - Precision (fake class)
    - Recall (fake class)
    - Equal Error Rate (EER)
    - Confusion Matrix

Generates visualizations:
    - ROC curve
    - Confusion matrix heatmap
    - Metrics summary table
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("metrics")


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate using ROC curve and root finding.

    The EER is the point where FAR (False Acceptance Rate) equals
    FRR (False Rejection Rate) on the ROC curve.

    Args:
        labels: Ground truth binary labels.
        scores: Prediction scores/probabilities.

    Returns:
        EER value (float).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    # Use interpolation to find where FPR == FNR
    try:
        eer_func = interp1d(fpr, fnr)
        # Find the intersection point (where fpr == fnr)
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    except (ValueError, RuntimeError):
        # Fallback: find the closest point
        diff = np.abs(fpr - fnr)
        idx = np.argmin(diff)
        eer = (fpr[idx] + fnr[idx]) / 2

    return float(eer)


def compute_all_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        labels: Ground truth labels (0=real, 1=fake).
        probs: Predicted probabilities.
        threshold: Classification threshold.

    Returns:
        Dictionary with all computed metrics.
    """
    labels = np.array(labels)
    probs = np.array(probs)
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds) * 100,
        "auc_roc": roc_auc_score(labels, probs),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_fake": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_fake": recall_score(labels, preds, pos_label=1, zero_division=0),
        "eer": compute_eer(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds),
        "threshold": threshold,
        "total_samples": len(labels),
        "real_count": int((labels == 0).sum()),
        "fake_count": int((labels == 1).sum()),
    }

    return metrics


def print_metrics_table(metrics: dict):
    """Print formatted metrics table to console/log."""
    logger.info("=" * 55)
    logger.info("       EVALUATION RESULTS")
    logger.info("=" * 55)
    logger.info(f"  Accuracy          : {metrics['accuracy']:.2f}%")
    logger.info(f"  AUC-ROC           : {metrics['auc_roc']:.4f}")
    logger.info(f"  F1-score (macro)  : {metrics['f1_macro']:.4f}")
    logger.info(f"  Precision (fake)  : {metrics['precision_fake']:.4f}")
    logger.info(f"  Recall (fake)     : {metrics['recall_fake']:.4f}")
    logger.info(f"  EER               : {metrics['eer']:.4f}")
    logger.info(f"  Threshold         : {metrics['threshold']}")
    logger.info(f"  Total Samples     : {metrics['total_samples']}")
    logger.info(f"  Real / Fake       : {metrics['real_count']} / {metrics['fake_count']}")
    logger.info("=" * 55)


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    class_names: list[str] = ["Real", "Fake"],
):
    """
    Plot and save confusion matrix heatmap.

    Args:
        cm: Confusion matrix array.
        save_path: Path to save the plot.
        class_names: Class labels.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": 16},
    )
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("Actual", fontsize=14)
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: str,
):
    """
    Plot and save ROC curve.

    Args:
        labels: Ground truth labels.
        probs: Predicted probabilities.
        save_path: Path to save the plot.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2563EB", linewidth=2.5, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2563EB")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve — Deepfake Detection", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved: {save_path}")


def save_metrics_table(metrics: dict, save_path: str):
    """Save metrics as a formatted text file."""
    with open(save_path, "w") as f:
        f.write("DEEPFAKE DETECTION — EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy          : {metrics['accuracy']:.2f}%\n")
        f.write(f"AUC-ROC           : {metrics['auc_roc']:.4f}\n")
        f.write(f"F1-score (macro)  : {metrics['f1_macro']:.4f}\n")
        f.write(f"Precision (fake)  : {metrics['precision_fake']:.4f}\n")
        f.write(f"Recall (fake)     : {metrics['recall_fake']:.4f}\n")
        f.write(f"EER               : {metrics['eer']:.4f}\n")
        f.write(f"Threshold         : {metrics['threshold']}\n\n")
        f.write(f"Total Samples     : {metrics['total_samples']}\n")
        f.write(f"Real / Fake       : {metrics['real_count']} / {metrics['fake_count']}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  {metrics['confusion_matrix']}\n")

    logger.info(f"Metrics table saved: {save_path}")
