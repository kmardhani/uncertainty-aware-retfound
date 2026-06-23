"""Tests for basic classification metrics."""

import pytest
import torch

from uncertainty_retfound.evaluation.metrics import (
    accuracy_score,
    auc_score_binary,
    balanced_accuracy_score_binary,
    classification_summary,
    confusion_matrix,
    f1_score_binary,
    per_class_accuracy,
    precision_score_binary,
    recall_score_binary,
    sensitivity_score_binary,
    specificity_score_binary,
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
    assert summary["precision"] == 1.0
    assert summary["recall"] == 0.5
    assert summary["sensitivity"] == 0.5
    assert summary["specificity"] == 1.0
    assert summary["f1"] == pytest.approx(2.0 / 3.0)
    assert summary["balanced_accuracy"] == 0.75
    assert summary["num_examples"] == 4
    assert summary["num_classes"] == 2


def test_precision_score_binary_returns_expected_value() -> None:
    predictions = torch.tensor([0, 1, 0, 0])
    labels = torch.tensor([0, 1, 1, 0])

    assert precision_score_binary(labels, predictions) == 1.0


def test_recall_and_sensitivity_scores_binary_return_expected_value() -> None:
    predictions = torch.tensor([0, 1, 0, 0])
    labels = torch.tensor([0, 1, 1, 0])

    assert recall_score_binary(labels, predictions) == 0.5
    assert sensitivity_score_binary(labels, predictions) == 0.5


def test_specificity_score_binary_returns_expected_value() -> None:
    predictions = torch.tensor([0, 1, 0, 0])
    labels = torch.tensor([0, 1, 1, 0])

    assert specificity_score_binary(labels, predictions) == 1.0


def test_f1_score_binary_returns_expected_value() -> None:
    predictions = torch.tensor([0, 1, 0, 0])
    labels = torch.tensor([0, 1, 1, 0])

    assert f1_score_binary(labels, predictions) == pytest.approx(2.0 / 3.0)


def test_balanced_accuracy_score_binary_returns_expected_value() -> None:
    predictions = torch.tensor([0, 1, 0, 0])
    labels = torch.tensor([0, 1, 1, 0])

    assert balanced_accuracy_score_binary(labels, predictions) == 0.75


def test_auc_score_binary_returns_expected_value() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    positive_scores = torch.tensor([0.1, 0.4, 0.35, 0.8])

    assert auc_score_binary(labels, positive_scores) == 0.75


def test_auc_score_binary_returns_none_when_only_one_class_is_present() -> None:
    labels = torch.tensor([1, 1, 1, 1])
    positive_scores = torch.tensor([0.8, 0.7, 0.6, 0.9])

    assert auc_score_binary(labels, positive_scores) is None


def test_zero_division_behavior_is_respected_for_binary_metrics() -> None:
    predictions = torch.tensor([0, 0, 0])
    labels = torch.tensor([1, 1, 1])

    assert precision_score_binary(labels, predictions, zero_division=1.0) == 1.0
    assert recall_score_binary(labels, predictions, zero_division=0.75) == 0.0
    assert specificity_score_binary(labels, predictions, zero_division=0.25) == 0.25
    assert f1_score_binary(labels, predictions, zero_division=0.5) == 0.0


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
