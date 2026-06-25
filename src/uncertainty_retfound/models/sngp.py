"""SNGP-style cached-feature heads with random Fourier features."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _apply_spectral_norm(linear: nn.Linear) -> nn.Module:
    """Apply spectral normalization using the available PyTorch API."""

    try:
        spectral_norm = torch.nn.utils.parametrizations.spectral_norm
    except AttributeError:
        spectral_norm = torch.nn.utils.spectral_norm
    return spectral_norm(linear)


class RandomFourierFeatureLayer(nn.Module):
    """Fixed random Fourier features approximating an RBF kernel."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive. Got: {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive. Got: {output_dim}")
        if kernel_scale <= 0.0:
            raise ValueError(f"kernel_scale must be positive. Got: {kernel_scale}")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.kernel_scale = float(kernel_scale)
        self.feature_scale = math.sqrt(2.0 / float(output_dim))

        random_weight = torch.randn(input_dim, output_dim, dtype=torch.float32) / kernel_scale
        random_bias = torch.rand(output_dim, dtype=torch.float32) * (2.0 * math.pi)
        self.register_buffer("random_weight", random_weight)
        self.register_buffer("random_bias", random_bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return random Fourier features for the given inputs."""

        projected = inputs @ self.random_weight + self.random_bias
        return self.feature_scale * torch.cos(projected)


class SNGPFeatureClassifier(nn.Module):
    """A lightweight SNGP-style classifier for frozen cached features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 0,
        rff_dim: int = 1024,
        kernel_scale: float = 1.0,
        ridge_penalty: float = 1.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive. Got: {input_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive. Got: {num_classes}")
        if hidden_dim < 0:
            raise ValueError(f"hidden_dim must be non-negative. Got: {hidden_dim}")
        if rff_dim <= 0:
            raise ValueError(f"rff_dim must be positive. Got: {rff_dim}")
        if ridge_penalty <= 0.0:
            raise ValueError(f"ridge_penalty must be positive. Got: {ridge_penalty}")

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.rff_dim = int(rff_dim)
        self.kernel_scale = float(kernel_scale)
        self.ridge_penalty = float(ridge_penalty)

        if self.hidden_dim > 0:
            self.projection = _apply_spectral_norm(nn.Linear(self.input_dim, self.hidden_dim))
            self.activation = nn.ReLU()
            rff_input_dim = self.hidden_dim
        else:
            self.projection = None
            self.activation = None
            rff_input_dim = self.input_dim

        self.random_features = RandomFourierFeatureLayer(
            input_dim=rff_input_dim,
            output_dim=self.rff_dim,
            kernel_scale=self.kernel_scale,
        )
        self.classifier = nn.Linear(self.rff_dim, self.num_classes)
        self.register_buffer(
            "precision_diag",
            torch.full((self.rff_dim,), self.ridge_penalty, dtype=torch.float32),
        )

    def project_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the optional spectral-normalized projection."""

        if self.projection is None or self.activation is None:
            return inputs
        return self.activation(self.projection(inputs))

    def random_feature_activations(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the fixed random-feature representation."""

        projected = self.project_features(inputs)
        return self.random_features(projected)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return classification logits."""

        random_features = self.random_feature_activations(inputs)
        return self.classifier(random_features)

    def reset_precision(self) -> None:
        """Reset the diagonal precision estimate to the ridge penalty."""

        self.precision_diag.fill_(self.ridge_penalty)

    def update_precision(self, random_features: torch.Tensor) -> None:
        """Accumulate diagonal precision statistics from random features."""

        if random_features.ndim != 2 or random_features.shape[1] != self.rff_dim:
            raise ValueError(
                f"random_features must have shape [N, {self.rff_dim}]. "
                f"Got: {tuple(random_features.shape)}"
            )
        self.precision_diag.add_(random_features.detach().square().sum(dim=0))

    def predictive_variance(self, random_features: torch.Tensor) -> torch.Tensor:
        """Return a diagonal precision-based variance proxy for each example."""

        clamped_precision = torch.clamp(self.precision_diag, min=1e-12)
        return (random_features.square() / clamped_precision.unsqueeze(0)).sum(dim=1)

    def predictive_statistics(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return logits, probabilities, and SNGP-style uncertainty statistics."""

        random_features = self.random_feature_activations(inputs)
        logits = self.classifier(random_features)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        clamped_probabilities = torch.clamp(probabilities, min=1e-12, max=1.0)
        predictive_entropy = -(
            clamped_probabilities * torch.log(clamped_probabilities)
        ).sum(dim=1)
        sngp_variance = self.predictive_variance(random_features)
        sngp_uncertainty = predictive_entropy + sngp_variance

        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "predictive_entropy": predictive_entropy,
            "sngp_variance": sngp_variance,
            "sngp_uncertainty": sngp_uncertainty,
            "random_features": random_features,
        }

    def to_serializable_config(self) -> dict[str, Any]:
        """Return simple hyperparameters useful for checkpoint metadata."""

        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "rff_dim": self.rff_dim,
            "kernel_scale": self.kernel_scale,
            "ridge_penalty": self.ridge_penalty,
        }
