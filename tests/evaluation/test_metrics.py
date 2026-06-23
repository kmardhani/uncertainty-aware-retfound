"""Tests for basic classification metrics."""

import pytest
import torch

from uncertainty_retfound.evaluation.metrics import (
    accuracy_score,
    classification_summary,
    confusion_matrix,
    per_class_accuracy,
)


def test_accuracy_score_returns_one_for_perfect_predictions() -> None:
    predictions = torch.tensor([0, 1, 1, 0])
    labels = torch.tensor([0, 1, 1, 0])

    assert accuracy_score(predictions, labels) == 1.0


def test_accuracy_score_returns_expected_partial_value() -> None:
    predictions = torch.tensor([0, 1, 0, 0])
    labels = torch.tensor([0, 1, 1, 0])

    assert accuracy_score(predictions, labels) == 0.75


def test_confusion_matrix_uses_rows_for_true_labels_and_columns_for_predictions() -> None:
    predictions = torch.tensor([0, 1, 0, 0])
    labels = torch.tensor([0, 1, 1, 0])

    matrix = confusion_matrix(predictions, labels, num_classes=2)

    assert torch.equal(matrix, torch.tensor([[2, 0], [1, 1]]))


def test_per_class_accuracy_returns_expected_values() -> None:
    predictions = torch.tensor([0, 1, 0, 0])
    labels = torch.tensor([0, 1, 1, 0])

    accuracies = per_class_accuracy(predictions, labels, num_classes=2)

    assert accuracies == {0: 1.0, 1: 0.5}


def test_per_class_accuracy_returns_none_for_missing_true_class() -> None:
    predictions = torch.tensor([0, 0, 0])
    labels = torch.tensor([0, 0, 0])

    accuracies = per_class_accuracy(predictions, labels, num_classes=2)

    assert accuracies == {0: 1.0, 1: None}


def test_classification_summary_includes_expected_keys_and_values() -> None:
    predictions = torch.tensor([0, 1, 0, 0])
    labels = torch.tensor([0, 1, 1, 0])

    summary = classification_summary(predictions, labels, num_classes=2)

    assert summary["accuracy"] == 0.75
    assert torch.equal(summary["confusion_matrix"], torch.tensor([[2, 0], [1, 1]]))
    assert summary["per_class_accuracy"] == {0: 1.0, 1: 0.5}
    assert summary["num_examples"] == 4
    assert summary["num_classes"] == 2


def test_accuracy_score_rejects_invalid_shapes() -> None:
    predictions = torch.tensor([[0, 1]])
    labels = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="1D tensors"):
        accuracy_score(predictions, labels)


def test_metrics_reject_empty_tensors() -> None:
    predictions = torch.tensor([], dtype=torch.int64)
    labels = torch.tensor([], dtype=torch.int64)

    with pytest.raises(ValueError, match="must not be empty"):
        accuracy_score(predictions, labels)


def test_confusion_matrix_rejects_out_of_range_class_values() -> None:
    predictions = torch.tensor([0, 2])
    labels = torch.tensor([0, 1])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        confusion_matrix(predictions, labels, num_classes=2)
