"""Basic classification metrics for baseline evaluation."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import roc_auc_score
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


def _validate_binary_labels_and_predictions(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    negative_label: int = 0,
    positive_label: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate binary classification labels and predictions."""

    validated_predictions, validated_labels = _validate_predictions_and_labels(y_pred, y_true)
    allowed_labels = {negative_label, positive_label}

    if any(int(value) not in allowed_labels for value in validated_predictions.tolist()):
        raise ValueError(
            "Binary predictions must only contain the configured negative and positive labels."
        )

    if any(int(value) not in allowed_labels for value in validated_labels.tolist()):
        raise ValueError(
            "Binary labels must only contain the configured negative and positive labels."
        )

    return validated_labels, validated_predictions


def _safe_divide(numerator: int, denominator: int, zero_division: float) -> float:
    """Return a safe float division with an explicit zero-division value."""

    if denominator == 0:
        return float(zero_division)

    return float(numerator / denominator)


def _validate_binary_scores(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    negative_label: int = 0,
    positive_label: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate binary labels and positive-class scores."""

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must be 1D tensors.")

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have matching shapes.")

    if y_true.numel() == 0:
        raise ValueError("y_true and y_score must not be empty.")

    validated_labels = y_true.to(dtype=torch.int64)
    allowed_labels = {negative_label, positive_label}

    if any(int(value) not in allowed_labels for value in validated_labels.tolist()):
        raise ValueError("Binary labels must only contain the configured negative and positive labels.")

    return validated_labels, y_score.to(dtype=torch.float32)


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


def precision_score_binary(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    positive_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """Return binary precision."""

    validated_labels, validated_predictions = _validate_binary_labels_and_predictions(
        y_true,
        y_pred,
        positive_label=positive_label,
        negative_label=0 if positive_label != 0 else 1,
    )
    true_positive = int(
        ((validated_predictions == positive_label) & (validated_labels == positive_label))
        .sum()
        .item()
    )
    predicted_positive = int((validated_predictions == positive_label).sum().item())
    return _safe_divide(true_positive, predicted_positive, zero_division)


def recall_score_binary(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    positive_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """Return binary recall."""

    validated_labels, validated_predictions = _validate_binary_labels_and_predictions(
        y_true,
        y_pred,
        positive_label=positive_label,
        negative_label=0 if positive_label != 0 else 1,
    )
    true_positive = int(
        ((validated_predictions == positive_label) & (validated_labels == positive_label))
        .sum()
        .item()
    )
    actual_positive = int((validated_labels == positive_label).sum().item())
    return _safe_divide(true_positive, actual_positive, zero_division)


def sensitivity_score_binary(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    positive_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """Return binary sensitivity, an alias for recall."""

    return recall_score_binary(
        y_true=y_true,
        y_pred=y_pred,
        positive_label=positive_label,
        zero_division=zero_division,
    )


def specificity_score_binary(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    negative_label: int = 0,
    positive_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """Return binary specificity."""

    validated_labels, validated_predictions = _validate_binary_labels_and_predictions(
        y_true,
        y_pred,
        negative_label=negative_label,
        positive_label=positive_label,
    )
    true_negative = int(
        ((validated_predictions == negative_label) & (validated_labels == negative_label))
        .sum()
        .item()
    )
    actual_negative = int((validated_labels == negative_label).sum().item())
    return _safe_divide(true_negative, actual_negative, zero_division)


def f1_score_binary(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    positive_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """Return binary F1 score."""

    precision = precision_score_binary(
        y_true=y_true,
        y_pred=y_pred,
        positive_label=positive_label,
        zero_division=zero_division,
    )
    recall = recall_score_binary(
        y_true=y_true,
        y_pred=y_pred,
        positive_label=positive_label,
        zero_division=zero_division,
    )

    if precision == 0.0 and recall == 0.0:
        return float(zero_division)

    return float((2.0 * precision * recall) / (precision + recall))


def balanced_accuracy_score_binary(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    negative_label: int = 0,
    positive_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """Return binary balanced accuracy."""

    sensitivity = sensitivity_score_binary(
        y_true=y_true,
        y_pred=y_pred,
        positive_label=positive_label,
        zero_division=zero_division,
    )
    specificity = specificity_score_binary(
        y_true=y_true,
        y_pred=y_pred,
        negative_label=negative_label,
        positive_label=positive_label,
        zero_division=zero_division,
    )
    return float((sensitivity + specificity) / 2.0)


def auc_score_binary(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    positive_label: int = 1,
) -> float | None:
    """Return binary ROC AUC from positive-class scores.

    Returns ``None`` when AUC is undefined because only one class is present in
    ``y_true``.
    """

    negative_label = 0 if positive_label != 0 else 1
    validated_labels, validated_scores = _validate_binary_scores(
        y_true,
        y_score,
        negative_label=negative_label,
        positive_label=positive_label,
    )

    if validated_labels.unique().numel() < 2:
        return None

    binary_labels = (validated_labels == positive_label).to(dtype=torch.int64)
    return float(roc_auc_score(binary_labels.tolist(), validated_scores.tolist()))


def classification_summary(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    positive_scores: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return a small classification summary for logging and testing."""

    validated_predictions, validated_labels = _validate_predictions_and_labels(
        predictions,
        labels,
    )
    matrix = confusion_matrix(validated_predictions, validated_labels, num_classes)

    summary: dict[str, Any] = {
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

    if num_classes == 2:
        summary["precision"] = precision_score_binary(validated_labels, validated_predictions)
        summary["recall"] = recall_score_binary(validated_labels, validated_predictions)
        summary["sensitivity"] = sensitivity_score_binary(validated_labels, validated_predictions)
        summary["specificity"] = specificity_score_binary(validated_labels, validated_predictions)
        summary["f1"] = f1_score_binary(validated_labels, validated_predictions)
        summary["balanced_accuracy"] = balanced_accuracy_score_binary(
            validated_labels,
            validated_predictions,
        )

        if positive_scores is not None:
            summary["auc"] = auc_score_binary(validated_labels, positive_scores)

    return summary
