"""Tests for SNGP-style cached-feature heads."""

from __future__ import annotations

import torch

from uncertainty_retfound.models.sngp import RandomFourierFeatureLayer, SNGPFeatureClassifier


def test_random_fourier_feature_layer_returns_expected_shape() -> None:
    layer = RandomFourierFeatureLayer(input_dim=4, output_dim=16, kernel_scale=2.0)
    inputs = torch.randn(3, 4)

    outputs = layer(inputs)

    assert outputs.shape == (3, 16)


def test_sngp_feature_classifier_returns_expected_logit_shape() -> None:
    model = SNGPFeatureClassifier(
        input_dim=4,
        num_classes=2,
        hidden_dim=8,
        rff_dim=16,
        kernel_scale=1.0,
        ridge_penalty=0.5,
    )
    inputs = torch.randn(5, 4)

    outputs = model(inputs)

    assert outputs.shape == (5, 2)


def test_sngp_predictive_statistics_are_finite() -> None:
    model = SNGPFeatureClassifier(
        input_dim=4,
        num_classes=2,
        hidden_dim=6,
        rff_dim=12,
        kernel_scale=1.5,
        ridge_penalty=1.0,
    )
    inputs = torch.randn(4, 4)
    random_features = model.random_feature_activations(inputs)
    model.reset_precision()
    model.update_precision(random_features)

    result = model.predictive_statistics(inputs)

    assert result["random_features"].shape == (4, 12)
    assert result["logits"].shape == (4, 2)
    assert result["probabilities"].shape == (4, 2)
    assert result["predictions"].shape == (4,)
    assert result["predictive_entropy"].shape == (4,)
    assert result["sngp_variance"].shape == (4,)
    assert result["sngp_uncertainty"].shape == (4,)
    assert torch.isfinite(result["predictive_entropy"]).all()
    assert torch.isfinite(result["sngp_variance"]).all()
    assert torch.isfinite(result["sngp_uncertainty"]).all()
