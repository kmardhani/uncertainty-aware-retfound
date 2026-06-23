"""Small training-step utilities for smoke-tested model training."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def train_one_batch(
    model: nn.Module,
    batch: dict[str, Any],
    optimizer: Optimizer,
    criterion: nn.Module | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run one optimization step for a single classification batch."""

    loss_fn = criterion or nn.CrossEntropyLoss()

    images = batch["image"]
    labels = batch["label"]

    if not isinstance(images, torch.Tensor):
        raise TypeError("batch['image'] must be a torch.Tensor.")

    if not isinstance(labels, torch.Tensor):
        raise TypeError("batch['label'] must be a torch.Tensor.")

    if device is not None:
        model = model.to(device)
        images = images.to(device)
        labels = labels.to(device)

    model.train()
    optimizer.zero_grad()

    logits = model(images)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "batch_size": int(labels.shape[0]),
    }
