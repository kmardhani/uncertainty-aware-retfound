"""Variational Bayesian linear heads for cached-feature experiments."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class BayesianLinear(nn.Module):
    """A Bayesian linear layer with diagonal Gaussian weight and bias posteriors."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
    ) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError(f"in_features must be positive. Got: {in_features}")

        if out_features <= 0:
            raise ValueError(f"out_features must be positive. Got: {out_features}")

        if prior_std <= 0.0:
            raise ValueError(f"prior_std must be positive. Got: {prior_std}")

        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.register_buffer("prior_std", torch.tensor(float(prior_std), dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize mean and rho parameters."""

        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.constant_(self.bias_rho, -5.0)

    @staticmethod
    def _rho_to_std(rho: torch.Tensor) -> torch.Tensor:
        """Convert unconstrained rho to positive std via softplus."""

        return F.softplus(rho)

    def sample_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample weights and biases with the reparameterization trick."""

        weight_std = self._rho_to_std(self.weight_rho)
        bias_std = self._rho_to_std(self.bias_rho)

        weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
        return weight, bias

    def forward(self, inputs: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Apply the Bayesian linear layer."""

        if sample:
            weight, bias = self.sample_parameters()
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(inputs, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """Return KL divergence against a zero-mean diagonal Gaussian prior."""

        weight_std = self._rho_to_std(self.weight_rho)
        bias_std = self._rho_to_std(self.bias_rho)
        prior_variance = self.prior_std.square()

        weight_kl = (
            torch.log(self.prior_std / weight_std)
            + (weight_std.square() + self.weight_mu.square()) / (2.0 * prior_variance)
            - 0.5
        ).sum()
        bias_kl = (
            torch.log(self.prior_std / bias_std)
            + (bias_std.square() + self.bias_mu.square()) / (2.0 * prior_variance)
            - 0.5
        ).sum()
        return weight_kl + bias_kl


class BayesianLinearClassifier(nn.Module):
    """A single Bayesian linear classification head."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        prior_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.classifier = BayesianLinear(
            in_features=feature_dim,
            out_features=num_classes,
            prior_std=prior_std,
        )

    def forward(self, inputs: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Return sampled or posterior-mean logits."""

        return self.classifier(inputs, sample=sample)

    def kl_divergence(self) -> torch.Tensor:
        """Return the KL divergence for the Bayesian head."""

        return self.classifier.kl_divergence()
