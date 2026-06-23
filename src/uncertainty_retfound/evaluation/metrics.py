"""Basic classification metrics for baseline evaluation."""

from __future__ import annotations

from typing import Any

import torch


def _validate_predictions_and_labels(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate prediction and label tensors for classification metrics."""

    if predictions.ndim != 1 or labels.ndim != 1:
        raise ValueError("Predictions and labels must be 1D tensors.")

    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have matching shapes.")

    if predictions.numel() == 0:
        raise ValueError("Predictions and labels must not be empty.")

    return predictions.to(dtype=torch.int64), labels.to(dtype=torch.int64)


def accuracy_score(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Return classification accuracy as a Python float."""

    validated_predictions, validated_labels = _validate_predictions_and_labels(
        predictions,
        labels,
    )
    return float((validated_predictions == validated_labels).float().mean().item())


def confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Return a confusion matrix with rows=true labels and columns=predictions."""

    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive. Got: {num_classes}")

    validated_predictions, validated_labels = _validate_predictions_and_labels(
        predictions,
        labels,
    )

    if (
        (validated_predictions < 0).any()
        or (validated_predictions >= num_classes).any()
        or (validated_labels < 0).any()
        or (validated_labels >= num_classes).any()
    ):
        raise ValueError(
            "Predictions and labels must be within the range "
            f"[0, {num_classes - 1}]."
        )

    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for true_label, predicted_label in zip(
        validated_labels.tolist(),
        validated_predictions.tolist(),
        strict=True,
    ):
        matrix[true_label, predicted_label] += 1

    return matrix


def per_class_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[int, float | None]:
    """Return accuracy for each true class."""

    matrix = confusion_matrix(predictions, labels, num_classes)
    accuracies: dict[int, float | None] = {}

    for class_index in range(num_classes):
        true_count = int(matrix[class_index].sum().item())
        if true_count == 0:
            accuracies[class_index] = None
            continue

        accuracies[class_index] = float(matrix[class_index, class_index].item() / true_count)

    return accuracies


def classification_summary(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[str, Any]:
    """Return a small classification summary for logging and testing."""

    validated_predictions, validated_labels = _validate_predictions_and_labels(
        predictions,
        labels,
    )
    matrix = confusion_matrix(validated_predictions, validated_labels, num_classes)

    return {
        "accuracy": accuracy_score(validated_predictions, validated_labels),
        "confusion_matrix": matrix,
        "per_class_accuracy": per_class_accuracy(
            validated_predictions,
            validated_labels,
            num_classes,
        ),
        "num_examples": int(validated_labels.numel()),
        "num_classes": int(num_classes),
    }
