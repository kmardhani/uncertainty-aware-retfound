"""Small reusable evaluation loop utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from uncertainty_retfound.evaluation.metrics import classification_summary


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module | None = None,
    device: torch.device | str | None = None,
    include_metrics: bool = False,
) -> dict[str, Any]:
    """Evaluate a model over a dataloader without gradient updates."""

    loss_fn = criterion or nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0
    num_examples = 0
    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []

    if device is not None:
        model = model.to(device)

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"]
            labels = batch["label"]

            if not isinstance(images, torch.Tensor):
                raise TypeError("batch['image'] must be a torch.Tensor.")

            if not isinstance(labels, torch.Tensor):
                raise TypeError("batch['label'] must be a torch.Tensor.")

            if device is not None:
                images = images.to(device)
                labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            num_batches += 1
            num_examples += batch_size

            logits_batches.append(logits.detach().cpu())
            label_batches.append(labels.detach().cpu())
            prediction_batches.append(predictions.detach().cpu())

    if num_batches == 0:
        raise ValueError("Cannot evaluate with an empty dataloader.")

    logits = torch.cat(logits_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    predictions = torch.cat(prediction_batches, dim=0)

    result: dict[str, Any] = {
        "loss": total_loss / num_examples,
        "num_batches": num_batches,
        "num_examples": num_examples,
        "logits": logits,
        "labels": labels,
        "predictions": predictions,
    }

    if include_metrics:
        result["metrics"] = classification_summary(
            predictions=predictions,
            labels=labels,
            num_classes=int(logits.shape[1]),
        )

    return result
