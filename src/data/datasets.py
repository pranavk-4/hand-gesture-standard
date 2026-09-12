"""Datasets, transforms, dataloaders, class weights.

Ported from hand_gesture_classifier/src/data/datasets.py with one
decoupling: instead of taking an AblationSettings object, functions take
two plain bools (use_heavy_augmentation, use_class_weights). The
baseline passes both from config; the HPO track passes per-trial
overrides without importing any ablation machinery.
"""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, use_heavy_augmentation: bool = False):
    train_ops = [
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ]
    if use_heavy_augmentation:
        train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))

    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.1),
    ]
    train_transform = transforms.Compose(train_ops)

    eval_transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def compute_class_weights(train_ds) -> torch.Tensor:
    labels = [label for _, label in train_ds.samples]
    counts = np.bincount(labels, minlength=len(train_ds.classes))
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(cfg, use_heavy_augmentation: bool | None = None,
                      use_class_weights: bool | None = None):
    from src.config import DATA_DIR
    if use_heavy_augmentation is None:
        use_heavy_augmentation = bool(cfg.training.augmentation)
    if use_class_weights is None:
        use_class_weights = bool(cfg.training.class_weighted_loss)

    processed_dir = DATA_DIR / "processed"
    train_tf, eval_tf = build_transforms(cfg.preprocessing.image_size, use_heavy_augmentation)

    train_ds = datasets.ImageFolder(processed_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(processed_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(processed_dir / "test", transform=eval_tf)

    expected = list(cfg.dataset.class_names)
    assert train_ds.classes == expected, (
        f"ImageFolder class order {train_ds.classes} != config class_names {expected}"
    )

    class_weights = compute_class_weights(train_ds) if use_class_weights else None

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,
                              num_workers=cfg.training.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False,
                            num_workers=cfg.training.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False,
                             num_workers=cfg.training.num_workers)
    return train_loader, val_loader, test_loader, class_weights
