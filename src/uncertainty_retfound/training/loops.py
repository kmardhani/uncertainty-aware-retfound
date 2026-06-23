"""Small reusable training loop utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from uncertainty_retfound.training.steps import train_one_batch


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    optimizer: Optimizer,
    criterion: nn.Module | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float | int]:
    """Train a model for one epoch over a dataloader."""

    total_loss = 0.0
    num_batches = 0
    num_examples = 0

    for batch in dataloader:
        result = train_one_batch(
            model=model,
            batch=batch,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        batch_size = int(result["batch_size"])
        total_loss += float(result["loss"]) * batch_size
        num_batches += 1
        num_examples += batch_size

    if num_batches == 0:
        raise ValueError("Cannot train for one epoch with an empty dataloader.")

    return {
        "loss": total_loss / num_examples,
        "num_batches": num_batches,
        "num_examples": num_examples,
    }
