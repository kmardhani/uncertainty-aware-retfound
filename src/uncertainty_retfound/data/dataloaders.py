"""Small dataloader helpers for retinal image experiments."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset


def create_dataloader(
    dataset: Dataset[Any],
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[Any]:
    """Create a simple PyTorch dataloader for a dataset."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
