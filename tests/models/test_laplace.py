"""Tests for diagonal Laplace utilities."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from uncertainty_retfound.models.laplace import (
    fit_diagonal_laplace_posterior,
    sample_laplace_logits,
)


class _FeatureDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.features = torch.tensor(
            [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.5, 0.25, 0.75]],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([1, 0, 1], dtype=torch.int64)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.features[index],
            "label": self.labels[index],
        }


def test_fit_diagonal_laplace_posterior_returns_expected_shapes() -> None:
    model = nn.Linear(3, 2)
    dataloader = DataLoader(_FeatureDataset(), batch_size=2, shuffle=False)

    posterior = fit_diagonal_laplace_posterior(
        model=model,
        dataloader=dataloader,
        prior_precision=1.0,
    )

    assert posterior.weight_mean.shape == (2, 3)
    assert posterior.weight_variance.shape == (2, 3)
    assert posterior.bias_mean.shape == (2,)
    assert posterior.bias_variance.shape == (2,)
    assert torch.all(posterior.weight_variance > 0.0)
    assert torch.all(posterior.bias_variance > 0.0)


def test_sample_laplace_logits_returns_expected_shape() -> None:
    model = nn.Linear(3, 2)
    dataloader = DataLoader(_FeatureDataset(), batch_size=2, shuffle=False)
    posterior = fit_diagonal_laplace_posterior(
        model=model,
        dataloader=dataloader,
        prior_precision=1.0,
    )
    features = torch.randn(4, 3)

    logits = sample_laplace_logits(
        posterior=posterior,
        features=features,
        num_samples=5,
    )

    assert logits.shape == (5, 4, 2)
