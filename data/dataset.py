"""
Dataset builder and preprocessing pipeline for deepfake detection.
Handles frame extraction, augmentation, and balanced dataset construction.
Dataset: 60,000 frames (224x224) — 30,000 real / 30,000 fake (~28.2 GB)
"""

import os
import cv2
import json
import random
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────
# Frame Extractor
# ──────────────────────────────────────────────

class FrameExtractor:
    """
    Extract frames from video files.
    Supports frame skipping for inference-speed optimization (→ 23.4 FPS).
    """

    def __init__(
        self,
        output_dir: str,
        img_size: Tuple[int, int] = (224, 224),
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.max_frames = max_frames

    def extract(self, video_path: str, label: str) -> List[str]:
        """Extract frames and save as JPEGs. Returns list of saved paths."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return []

        video_name = Path(video_path).stem
        save_dir = self.output_dir / label / video_name
        save_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_skip == 0:
                frame_resized = cv2.resize(frame, self.img_size)
                out_path = save_dir / f"frame_{saved_count:05d}.jpg"
                cv2.imwrite(str(out_path), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved.append(str(out_path))
                saved_count += 1

                if self.max_frames and saved_count >= self.max_frames:
                    break

            frame_idx += 1

        cap.release()
        logger.info(f"Extracted {saved_count} frames from {video_name}")
        return saved

    def extract_batch(self, video_dir: str, label: str) -> List[str]:
        """Extract frames from all videos in a directory."""
        all_paths = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            for vp in Path(video_dir).glob(ext):
                all_paths.extend(self.extract(str(vp), label))
        return all_paths


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for deepfake detection.

    Expects directory structure:
        root/
            real/  *.jpg
            fake/  *.jpg

    Labels: 0 = real, 1 = fake
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",           # "train" | "val" | "test"
        img_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        manifest_path: Optional[str] = None,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform or (
            get_train_transforms(img_size) if split == "train"
            else get_val_transforms(img_size)
        )

        if manifest_path and Path(manifest_path).exists():
            self.samples = self._load_manifest(manifest_path)
        else:
            self.samples = self._scan_directory()

        # Reproducible shuffle
        random.seed(seed)
        random.shuffle(self.samples)

        logger.info(
            f"[{split.upper()}] {len(self.samples)} samples | "
            f"real={sum(1 for _,l in self.samples if l==0)} | "
            f"fake={sum(1 for _,l in self.samples if l==1)}"
        )

    # ── Internal helpers ──────────────────────

    def _scan_directory(self) -> List[Tuple[str, int]]:
        samples = []
        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            label_dir = self.root_dir / label_name
            if not label_dir.exists():
                logger.warning(f"Directory not found: {label_dir}")
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for fp in label_dir.rglob(ext):
                    samples.append((str(fp), label_idx))
        return samples

    def _load_manifest(self, path: str) -> List[Tuple[str, int]]:
        with open(path) as f:
            data = json.load(f)
        return [(item["path"], item["label"]) for item in data]

    def save_manifest(self, path: str) -> None:
        manifest = [{"path": p, "label": l} for p, l in self.samples]
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest saved to {path}")

    # ── Dataset API ───────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        return tensor, label

    def get_class_weights(self) -> torch.Tensor:
        """For WeightedRandomSampler — balances imbalanced datasets."""
        labels = [l for _, l in self.samples]
        counts = np.bincount(labels)
        weights = 1.0 / counts
        sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float)
        return sample_weights


# ──────────────────────────────────────────────
# DataLoader Factory
# ──────────────────────────────────────────────

def build_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    use_weighted_sampler: bool = False,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Args:
        data_root: root directory with train/, val/, test/ subdirs
        batch_size: default 32 (matches training config)
        num_workers: parallel data loading workers
        img_size: default 224 (ResNet-50 input)
        use_weighted_sampler: oversample minority class in training
        pin_memory: faster GPU transfer

    Returns:
        dict with keys "train", "val", "test"
    """
    loaders = {}

    for split in ("train", "val", "test"):
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            logger.warning(f"Split directory missing: {split_dir}")
            continue

        ds = DeepfakeDataset(root_dir=split_dir, split=split, img_size=img_size)

        sampler = None
        shuffle = (split == "train")

        if split == "train" and use_weighted_sampler:
            sample_weights = ds.get_class_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False  # mutually exclusive with sampler

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )

    return loaders


# ──────────────────────────────────────────────
# Dataset Statistics
# ──────────────────────────────────────────────

def compute_dataset_stats(loader: DataLoader, n_batches: int = 50) -> Dict:
    """Compute mean/std over a subset of batches for normalization tuning."""
    means, stds = [], []
    for i, (imgs, _) in enumerate(loader):
        if i >= n_batches:
            break
        means.append(imgs.mean(dim=[0, 2, 3]))
        stds.append(imgs.std(dim=[0, 2, 3]))
    mean = torch.stack(means).mean(0)
    std = torch.stack(stds).mean(0)
    return {"mean": mean.tolist(), "std": std.tolist()}


if __name__ == "__main__":
    # Quick smoke-test with synthetic data
    import tempfile, shutil

    tmp = tempfile.mkdtemp()
    for split in ("train", "val", "test"):
        for cls in ("real", "fake"):
            d = Path(tmp) / split / cls
            d.mkdir(parents=True)
            for i in range(10):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img.save(d / f"img_{i:03d}.jpg")

    loaders = build_dataloaders(tmp, batch_size=4, num_workers=0)
    for split, loader in loaders.items():
        imgs, labels = next(iter(loader))
        print(f"{split}: batch {imgs.shape}, labels {labels}")

    shutil.rmtree(tmp)
    print("Dataset module OK")
