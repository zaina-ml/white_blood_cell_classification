"""
dataset.py
Data pipeline: dataset construction, transforms, weighted sampler,
and DataLoader factory.

CPU transforms handle loading and type conversion; GPU transforms
(augmentation, normalisation) run on-device inside the training loop
for maximum throughput.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from config import cfg

cpu_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8),
])

label_transform = v2.Compose([
    v2.ToDtype(torch.long),
])


def build_gpu_transforms(device: torch.device):
    size = (cfg.image_size, cfg.image_size)

    train_transform = nn.Sequential(
        v2.Resize(size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(15),
        v2.RandomAdjustSharpness(sharpness_factor=2.0),
        v2.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
    ).to(device)

    eval_transform = nn.Sequential(
        v2.Resize(size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
    ).to(device)

    return train_transform, eval_transform



def build_weighted_sampler(dataset: ImageFolder):
    targets       = dataset.targets
    class_counts  = np.bincount(targets)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label].item() for label in targets]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )



def build_datasets(data_root):
    import pathlib
    data_root = pathlib.Path(data_root)

    train_dataset = ImageFolder(
        root=data_root / cfg.train_subdir,
        transform=cpu_transform,
        target_transform=label_transform,
    )
    test_dataset = ImageFolder(
        root=data_root / cfg.test_a_subdir,
        transform=cpu_transform,
        target_transform=label_transform,
    )
    return train_dataset, test_dataset


def build_dataloaders(
    train_dataset: ImageFolder,
    test_dataset:  ImageFolder,
):
    sampler = build_weighted_sampler(train_dataset)

    loader_kwargs = dict(
        pin_memory=cfg.pin_memory,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    return train_loader, test_loader