"""
PyTorch Dataset for loading preprocessed deepfake detection data.

Loads face-cropped images from the structured dataset directory and applies
augmentations for training.
"""

import os
import glob
import csv
import numpy as np
from PIL import Image
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config
from utils.logger import get_logger

logger = get_logger("video_dataset")


def get_train_transforms(cfg: Config) -> A.Compose:
    """Build training augmentation pipeline."""
    return A.Compose([
        A.HorizontalFlip(p=cfg.aug_horizontal_flip_p),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1,
            p=cfg.aug_color_jitter_p,
        ),
        A.RandomResizedCrop(
            size=(cfg.face_size, cfg.face_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=cfg.aug_random_crop_p,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=cfg.aug_gaussian_blur_p),
        A.ImageCompression(
            quality_lower=cfg.aug_jpeg_quality_range[0],
            quality_upper=cfg.aug_jpeg_quality_range[1],
            p=cfg.aug_jpeg_compression_p,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(cfg: Config) -> A.Compose:
    """Build validation/test augmentation pipeline (no augmentation, just normalize)."""
    return A.Compose([
        A.Resize(cfg.face_size, cfg.face_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


class DeepfakeDataset(Dataset):
    """
    Dataset for deepfake detection.

    Loads face-cropped images from a structured directory:
        root/{split}/real/*.png
        root/{split}/fake/*.png

    Labels:
        0 = real
        1 = fake
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        extensions: tuple[str, ...] = ("png", "jpg", "jpeg"),
    ):
        """
        Args:
            root_dir: Root directory containing split folders.
            split: One of 'train', 'val', 'test'.
            transform: Albumentations transform pipeline.
            extensions: Image file extensions to include.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.image_paths = []
        self.labels = []

        split_dir = os.path.join(root_dir, split)

        # Load real images (label = 0)
        real_dir = os.path.join(split_dir, "real")
        if os.path.isdir(real_dir):
            for ext in extensions:
                paths = sorted(glob.glob(os.path.join(real_dir, f"*.{ext}")))
                self.image_paths.extend(paths)
                self.labels.extend([0] * len(paths))

        # Load fake images (label = 1)
        fake_dir = os.path.join(split_dir, "fake")
        if os.path.isdir(fake_dir):
            for ext in extensions:
                paths = sorted(glob.glob(os.path.join(fake_dir, f"*.{ext}")))
                self.image_paths.extend(paths)
                self.labels.extend([1] * len(paths))

        logger.info(
            f"Loaded {split} split: "
            f"{sum(1 for l in self.labels if l == 0)} real, "
            f"{sum(1 for l in self.labels if l == 1)} fake, "
            f"{len(self.labels)} total"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (image_tensor [C, H, W], label_tensor [1]).
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label_tensor = torch.tensor([label], dtype=torch.float32)

        return image, label_tensor

    def get_path(self, idx: int) -> str:
        """Get the file path for a specific index."""
        return self.image_paths[idx]


class CSVDataset(Dataset):
    """
    Dataset for deepfake detection reading from a CSV file.
    Expects CSV format: image_path, label
    - label: 0 for real, 1 for fake
    """

    def __init__(
        self,
        csv_path: str,
        transform: Optional[A.Compose] = None,
    ):
        self.csv_path = csv_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                try:
                    self.labels.append(int(header[1]))
                    self.image_paths.append(header[0])
                except ValueError:
                    pass  # Header row
            
            for row in reader:
                if len(row) >= 2:
                    self.image_paths.append(row[0])
                    self.labels.append(int(row[1]))

        logger.info(
            f"Loaded CSV split from {csv_path}: "
            f"{sum(1 for l in self.labels if l == 0)} real, "
            f"{sum(1 for l in self.labels if l == 1)} fake, "
            f"{len(self.labels)} total"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label_tensor = torch.tensor([label], dtype=torch.float32)

        return image, label_tensor

    def get_path(self, idx: int) -> str:
        """Get the file path for a specific index."""
        return self.image_paths[idx]


def create_dataloaders(cfg: Config) -> dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        cfg: Configuration object.

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders.
    """
    train_transform = get_train_transforms(cfg)
    val_transform = get_val_transforms(cfg)

    datasets = {}
    
    if hasattr(cfg, 'train_csv') and cfg.train_csv:
        datasets["train"] = CSVDataset(cfg.train_csv, train_transform)
    else:
        datasets["train"] = DeepfakeDataset(cfg.frames_dir, "train", train_transform)
        
    if hasattr(cfg, 'val_csv') and cfg.val_csv:
        datasets["val"] = CSVDataset(cfg.val_csv, val_transform)
    else:
        datasets["val"] = DeepfakeDataset(cfg.frames_dir, "val", val_transform)
        
    if hasattr(cfg, 'test_csv') and cfg.test_csv:
        datasets["test"] = CSVDataset(cfg.test_csv, val_transform)
    else:
        datasets["test"] = DeepfakeDataset(cfg.frames_dir, "test", val_transform)

    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=(split == "train"),
        )

    return loaders
