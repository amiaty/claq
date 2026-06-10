"""CIFAR-10 data helpers for the clean CLAQ repo."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader


def get_cifar10_datasets(transform, root: str | Path) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    root = str(root)
    train_ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    return train_ds, test_ds


def get_cifar10_loaders(
    transform,
    root: str | Path,
    batch_size: int,
    num_workers: int,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_ds, test_ds = get_cifar10_datasets(transform=transform, root=root)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def get_raw_cifar10_dataset(root: str | Path, train: bool = False):
    return torchvision.datasets.CIFAR10(root=str(root), train=train, download=True)
