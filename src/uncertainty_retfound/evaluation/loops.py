"""Small reusable evaluation loop utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from uncertainty_retfound.evaluation.metrics import classification_summary


def _extract_batch_string_values(
    batch: Mapping[str, object],
    key: str,
) -> list[str]:
    """Extract a list of string-like values from a batch mapping."""

    raw_values = batch.get(key)
    if raw_values is None:
        return []

    if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
        raise TypeError(f"batch['{key}'] must be a list or tuple of strings.")

    sequence_values = cast(Sequence[object], raw_values)
    extracted_values: list[str] = []
    for raw_value in sequence_values:
        extracted_values.append(str(raw_value))

    return extracted_values


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
    image_path_values: list[str] = []
    id_code_values: list[str] = []
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
            if not isinstance(batch, Mapping):
                raise TypeError("Each dataloader batch must be a mapping.")

            batch_mapping = cast(Mapping[str, object], batch)
            images_object: object = batch_mapping["image"]
            labels_object: object = batch_mapping["label"]

            if not isinstance(images_object, torch.Tensor):
                raise TypeError("batch['image'] must be a torch.Tensor.")

            if not isinstance(labels_object, torch.Tensor):
                raise TypeError("batch['label'] must be a torch.Tensor.")

            images = images_object
            labels = labels_object

            image_path_values.extend(_extract_batch_string_values(batch_mapping, "image_path"))
            id_code_values.extend(_extract_batch_string_values(batch_mapping, "id_code"))

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
    probabilities: torch.Tensor | None = None

    if logits.ndim == 2 and logits.shape[1] == 2:
        probabilities = torch.softmax(logits, dim=1)
        positive_scores = probabilities[:, 1]

    result: dict[str, Any] = {
        "loss": total_loss / num_examples,
        "num_batches": num_batches,
        "num_examples": num_examples,
        "logits": logits,
        "labels": labels,
        "predictions": predictions,
    }

    if logits.ndim == 2:
        result["probabilities"] = torch.softmax(logits, dim=1)

    if image_path_values:
        result["image_paths"] = image_path_values

    if id_code_values:
        result["id_codes"] = id_code_values

    if include_metrics:
        result["metrics"] = classification_summary(
            predictions=predictions,
            labels=labels,
            num_classes=int(logits.shape[1]),
            positive_scores=positive_scores,
            probabilities=probabilities,
            logits=logits if logits.ndim == 2 and logits.shape[1] == 2 else None,
        )

    if include_batch_history:
        result["batch_history"] = batch_history

    return result
