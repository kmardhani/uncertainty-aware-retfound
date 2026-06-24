"""Diagonal Laplace approximations for cached-feature linear heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class DiagonalLinearLaplacePosterior:
    """Diagonal Gaussian posterior for a linear classifier."""

    weight_mean: torch.Tensor
    weight_variance: torch.Tensor
    bias_mean: torch.Tensor
    bias_variance: torch.Tensor
    prior_precision: float

    def sample_parameters(
        self,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample linear parameters from the diagonal posterior."""

        weight_mean = self.weight_mean
        weight_variance = self.weight_variance
        bias_mean = self.bias_mean
        bias_variance = self.bias_variance

        if device is not None:
            weight_mean = weight_mean.to(device)
            weight_variance = weight_variance.to(device)
            bias_mean = bias_mean.to(device)
            bias_variance = bias_variance.to(device)

        weight = weight_mean + torch.sqrt(weight_variance) * torch.randn_like(weight_mean)
        bias = bias_mean + torch.sqrt(bias_variance) * torch.randn_like(bias_mean)
        return weight, bias

    def to_dict(self) -> dict[str, Any]:
        """Serialize the posterior to a torch-saveable mapping."""

        return {
            "weight_mean": self.weight_mean,
            "weight_variance": self.weight_variance,
            "bias_mean": self.bias_mean,
            "bias_variance": self.bias_variance,
            "prior_precision": float(self.prior_precision),
        }


def fit_diagonal_laplace_posterior(
    model: nn.Linear,
    dataloader: DataLoader[Any],
    prior_precision: float,
    device: torch.device | str | None = None,
) -> DiagonalLinearLaplacePosterior:
    """Fit a diagonal Laplace posterior around a trained linear softmax head."""

    if prior_precision <= 0.0:
        raise ValueError(f"prior_precision must be positive. Got: {prior_precision}")

    if device is not None:
        model = model.to(device)

    model.eval()

    weight_curvature = torch.zeros_like(model.weight, device=model.weight.device)
    bias_curvature = torch.zeros_like(model.bias, device=model.bias.device)

    with torch.no_grad():
        for batch in dataloader:
            features = batch["image"]
            if not isinstance(features, torch.Tensor):
                raise TypeError("batch['image'] must be a torch.Tensor.")

            if device is not None:
                features = features.to(device)

            logits = model(features)
            probabilities = torch.softmax(logits, dim=1)
            class_curvature = probabilities * (1.0 - probabilities)
            feature_squares = features.square()

            weight_curvature += class_curvature.transpose(0, 1) @ feature_squares
            bias_curvature += class_curvature.sum(dim=0)

    weight_variance = 1.0 / (weight_curvature + prior_precision)
    bias_variance = 1.0 / (bias_curvature + prior_precision)

    return DiagonalLinearLaplacePosterior(
        weight_mean=model.weight.detach().cpu().clone(),
        weight_variance=weight_variance.detach().cpu().clone(),
        bias_mean=model.bias.detach().cpu().clone(),
        bias_variance=bias_variance.detach().cpu().clone(),
        prior_precision=float(prior_precision),
    )


def sample_laplace_logits(
    posterior: DiagonalLinearLaplacePosterior,
    features: torch.Tensor,
    num_samples: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample Monte Carlo logits from a diagonal Laplace posterior."""

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive. Got: {num_samples}")

    logits_samples: list[torch.Tensor] = []
    for _ in range(num_samples):
        sampled_weight, sampled_bias = posterior.sample_parameters(device=device)
        logits_samples.append(F.linear(features, sampled_weight, sampled_bias))

    return torch.stack(logits_samples, dim=0)
