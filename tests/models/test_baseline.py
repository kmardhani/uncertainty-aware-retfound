"""Tests for small baseline image models."""

import torch

from uncertainty_retfound.models.baseline import SmallCNNClassifier


def test_small_cnn_classifier_returns_expected_logit_shape() -> None:
    model = SmallCNNClassifier(num_classes=2)
    inputs = torch.randn(4, 3, 32, 32)

    outputs = model(inputs)

    assert outputs.shape == (4, 2)
