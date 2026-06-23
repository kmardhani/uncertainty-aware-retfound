"""Small reusable evaluation loop utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from uncertainty_retfound.evaluation.metrics import classification_summary


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module | None = None,
    device: torch.device | str | None = None,
    include_metrics: bool = False,
    show_progress: bool = False,
    progress_description: str | None = None,
    include_batch_history: bool = False,
) -> dict[str, Any]:
    """Evaluate a model over a dataloader without gradient updates."""

    loss_fn = criterion or nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0
    num_examples = 0
    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []
    batch_history: list[dict[str, float | int]] = []

    if device is not None:
        model = model.to(device)

    model.eval()

    dataloader_iterator = tqdm(
        dataloader,
        desc=progress_description,
        disable=not show_progress,
        leave=False,
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader_iterator, start=1):
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
            batch_accuracy = float((predictions == labels).float().mean().item())

            batch_size = int(labels.shape[0])
            batch_loss = float(loss.item())

            total_loss += batch_loss * batch_size
            num_batches += 1
            num_examples += batch_size

            logits_batches.append(logits.detach().cpu())
            label_batches.append(labels.detach().cpu())
            prediction_batches.append(predictions.detach().cpu())

            if include_batch_history:
                batch_history.append(
                    {
                        "batch": batch_index,
                        "loss": batch_loss,
                        "accuracy": batch_accuracy,
                        "num_examples": batch_size,
                    }
                )

            if show_progress:
                dataloader_iterator.set_postfix_str(
                    f"loss={batch_loss:.4f}, accuracy={batch_accuracy:.4f}"
                )

    if num_batches == 0:
        raise ValueError("Cannot evaluate with an empty dataloader.")

    logits = torch.cat(logits_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    predictions = torch.cat(prediction_batches, dim=0)
    positive_scores: torch.Tensor | None = None

    if logits.ndim == 2 and logits.shape[1] == 2:
        positive_scores = torch.softmax(logits, dim=1)[:, 1]

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
            positive_scores=positive_scores,
        )

    if include_batch_history:
        result["batch_history"] = batch_history

    return result
