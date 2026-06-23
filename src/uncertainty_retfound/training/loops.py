"""Small reusable training loop utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from uncertainty_retfound.training.steps import train_one_batch


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    optimizer: Optimizer,
    criterion: nn.Module | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = False,
    progress_description: str | None = None,
    include_batch_history: bool = False,
) -> dict[str, Any]:
    """Train a model for one epoch over a dataloader."""

    total_loss = 0.0
    num_batches = 0
    num_examples = 0
    batch_history: list[dict[str, float | int]] = []

    dataloader_iterator = tqdm(
        dataloader,
        desc=progress_description,
        disable=not show_progress,
        leave=False,
    )

    for batch_index, batch in enumerate(dataloader_iterator, start=1):
        result = train_one_batch(
            model=model,
            batch=batch,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        batch_size = int(result["batch_size"])
        batch_loss = float(result["loss"])

        total_loss += float(result["loss"]) * batch_size
        num_batches += 1
        num_examples += batch_size

        if include_batch_history:
            batch_history.append(
                {
                    "batch": batch_index,
                    "loss": batch_loss,
                    "num_examples": batch_size,
                }
            )

        if show_progress:
            dataloader_iterator.set_postfix_str(f"loss={batch_loss:.4f}")

    if num_batches == 0:
        raise ValueError("Cannot train for one epoch with an empty dataloader.")

    result: dict[str, Any] = {
        "loss": total_loss / num_examples,
        "num_batches": num_batches,
        "num_examples": num_examples,
    }

    if include_batch_history:
        result["batch_history"] = batch_history

    return result
