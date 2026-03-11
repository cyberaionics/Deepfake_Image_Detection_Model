"""
Face Cropper using MTCNN for face detection, alignment, and cropping.

Processes extracted frames to detect faces, align them, crop, and resize
to a consistent 224×224 resolution for model input.
"""

import os
import cv2
import glob
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("face_cropper")


class FaceCropper:
    """
    Face detection, alignment, and cropping using MTCNN.

    Attributes:
        face_size: Output face image size (default: 224).
        margin: Margin around detected face bounding box.
        device: Torch device for MTCNN.
    """

    def __init__(self, face_size: int = 224, margin: int = 20, device: str = "cpu"):
        from facenet_pytorch import MTCNN

        self.face_size = face_size
        self.margin = margin
        self.device = device

        self.detector = MTCNN(
            image_size=face_size,
            margin=margin,
            keep_all=False,         # Return only the largest face
            select_largest=True,
            post_process=False,     # Return PIL image directly
            device=device,
        )

    def detect_and_crop(self, image_path: str) -> np.ndarray | None:
        """
        Detect, align, and crop face from an image.

        Args:
            image_path: Path to the input image.

        Returns:
            Cropped face as numpy array (H, W, C) in BGR, or None if no face detected.
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Cannot open image {image_path}: {e}")
            return None

        # Detect face — MTCNN returns aligned face tensor
        face = self.detector(img)

        if face is None:
            return None

        # Convert tensor (C, H, W) → numpy (H, W, C)
        face_np = face.permute(1, 2, 0).numpy().astype(np.uint8)

        # Convert RGB → BGR for cv2 saving
        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

        # Ensure correct size
        if face_bgr.shape[0] != self.face_size or face_bgr.shape[1] != self.face_size:
            face_bgr = cv2.resize(face_bgr, (self.face_size, self.face_size))

        return face_bgr

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg"),
    ) -> dict:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing input images.
            output_dir: Directory to save cropped faces.
            extensions: Image file extensions to process.

        Returns:
            Dictionary with counts of processed, skipped, and failed images.
        """
        os.makedirs(output_dir, exist_ok=True)

        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.sort()

        stats = {"processed": 0, "no_face": 0, "failed": 0}

        for img_path in tqdm(image_paths, desc=f"Cropping faces in {Path(input_dir).name}"):
            face = self.detect_and_crop(img_path)

            if face is None:
                stats["no_face"] += 1
                continue

            filename = Path(img_path).name
            save_path = os.path.join(output_dir, filename)

            try:
                cv2.imwrite(save_path, face)
                stats["processed"] += 1
            except Exception as e:
                logger.warning(f"Failed to save {save_path}: {e}")
                stats["failed"] += 1

        return stats


def process_dataset_splits(
    frames_root: str,
    output_root: str,
    face_size: int = 224,
    margin: int = 20,
    device: str = "cpu",
):
    """
    Process the entire dataset split structure.

    Expected input structure:
        frames_root/
            train/real/  train/fake/
            val/real/    val/fake/
            test/real/   test/fake/

    Args:
        frames_root: Root directory with raw extracted frames.
        output_root: Root directory for cropped face outputs.
        face_size: Target face crop size.
        margin: Margin around face bounding box.
        device: Device for MTCNN.
    """
    cropper = FaceCropper(face_size=face_size, margin=margin, device=device)

    splits = ["train", "val", "test"]
    labels = ["real", "fake"]

    total_stats = {"processed": 0, "no_face": 0, "failed": 0}

    for split in splits:
        for label in labels:
            input_dir = os.path.join(frames_root, split, label)
            output_dir = os.path.join(output_root, split, label)

            if not os.path.isdir(input_dir):
                logger.info(f"Skipping (not found): {input_dir}")
                continue

            logger.info(f"Processing: {split}/{label}")
            stats = cropper.process_directory(input_dir, output_dir)

            for k in total_stats:
                total_stats[k] += stats[k]

            logger.info(
                f"  {split}/{label}: "
                f"processed={stats['processed']}, "
                f"no_face={stats['no_face']}, "
                f"failed={stats['failed']}"
            )

    logger.info(f"Total: {total_stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and crop faces from extracted frames")
    parser.add_argument("--input", type=str, required=True, help="Root dir with frames (train/val/test)")
    parser.add_argument("--output", type=str, required=True, help="Output dir for cropped faces")
    parser.add_argument("--face_size", type=int, default=224)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    process_dataset_splits(
        frames_root=args.input,
        output_root=args.output,
        face_size=args.face_size,
        margin=args.margin,
        device=args.device,
    )
    logger.info("Face cropping complete!")
