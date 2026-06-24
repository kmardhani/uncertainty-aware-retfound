"""Tests for Bayesian linear heads."""

from __future__ import annotations

import torch

from uncertainty_retfound.models.bayesian import BayesianLinear, BayesianLinearClassifier


def test_bayesian_linear_returns_expected_shape() -> None:
    layer = BayesianLinear(in_features=4, out_features=2, prior_std=1.0)
    inputs = torch.randn(3, 4)

    outputs = layer(inputs, sample=True)

    assert outputs.shape == (3, 2)


def test_bayesian_linear_sampling_differs_from_posterior_mean() -> None:
    torch.manual_seed(42)
    layer = BayesianLinear(in_features=4, out_features=2, prior_std=1.0)
    inputs = torch.randn(2, 4)

    sampled_outputs = layer(inputs, sample=True)
    mean_outputs = layer(inputs, sample=False)

    assert not torch.allclose(sampled_outputs, mean_outputs)


def test_bayesian_linear_kl_divergence_is_finite_and_positive() -> None:
    layer = BayesianLinear(in_features=4, out_features=2, prior_std=1.0)

    kl_value = layer.kl_divergence()

    assert torch.isfinite(kl_value)
    assert float(kl_value.item()) >= 0.0


def test_bayesian_linear_classifier_returns_expected_logit_shape() -> None:
    model = BayesianLinearClassifier(feature_dim=4, num_classes=2, prior_std=1.0)
    inputs = torch.randn(5, 4)

    outputs = model(inputs, sample=True)

    assert outputs.shape == (5, 2)
