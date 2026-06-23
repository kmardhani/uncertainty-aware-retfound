"""Tests for RETFound-style wrapper models."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from uncertainty_retfound.models.retfound import (
    FrozenEncoderClassifier,
    build_retfound_linear_classifier,
)


class FakeTensorEncoder(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3, feature_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = inputs.mean(dim=(2, 3))
        return self.proj(pooled)


class FakeTokenEncoder(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3, feature_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = inputs.mean(dim=(2, 3))
        cls_token = self.proj(pooled)
        other_token = cls_token + 1.0
        return torch.stack([cls_token, other_token], dim=1)


class FakeDictFeaturesEncoder(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3, feature_dim)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = inputs.mean(dim=(2, 3))
        return {"features": self.proj(pooled)}


class FakeLastHiddenStateEncoder(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3, feature_dim)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = inputs.mean(dim=(2, 3))
        cls_token = self.proj(pooled)
        other_token = cls_token + 1.0
        return {"last_hidden_state": torch.stack([cls_token, other_token], dim=1)}


class FakeTupleEncoder(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3, feature_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor]:
        pooled = inputs.mean(dim=(2, 3))
        return (self.proj(pooled),)


class UnsupportedEncoder(nn.Module):
    def forward(self, inputs: torch.Tensor) -> str:
        return "unsupported"


def test_frozen_encoder_classifier_returns_expected_logit_shape() -> None:
    model = FrozenEncoderClassifier(
        encoder=FakeTensorEncoder(feature_dim=8),
        feature_dim=8,
        num_classes=2,
    )
    inputs = torch.randn(4, 3, 32, 32)

    outputs = model(inputs)

    assert outputs.shape == (4, 2)


def test_frozen_encoder_classifier_freezes_encoder_parameters() -> None:
    encoder = FakeTensorEncoder(feature_dim=8)

    FrozenEncoderClassifier(
        encoder=encoder,
        feature_dim=8,
        num_classes=2,
        freeze_encoder=True,
    )

    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_frozen_encoder_classifier_can_leave_encoder_trainable() -> None:
    encoder = FakeTensorEncoder(feature_dim=8)

    FrozenEncoderClassifier(
        encoder=encoder,
        feature_dim=8,
        num_classes=2,
        freeze_encoder=False,
    )

    assert all(parameter.requires_grad for parameter in encoder.parameters())


def test_frozen_encoder_classifier_supports_token_outputs() -> None:
    model = FrozenEncoderClassifier(
        encoder=FakeTokenEncoder(feature_dim=8),
        feature_dim=8,
        num_classes=2,
    )

    outputs = model(torch.randn(4, 3, 32, 32))

    assert outputs.shape == (4, 2)


def test_frozen_encoder_classifier_supports_dict_features_output() -> None:
    model = FrozenEncoderClassifier(
        encoder=FakeDictFeaturesEncoder(feature_dim=8),
        feature_dim=8,
        num_classes=2,
    )

    outputs = model(torch.randn(4, 3, 32, 32))

    assert outputs.shape == (4, 2)


def test_frozen_encoder_classifier_supports_last_hidden_state_output() -> None:
    model = FrozenEncoderClassifier(
        encoder=FakeLastHiddenStateEncoder(feature_dim=8),
        feature_dim=8,
        num_classes=2,
    )

    outputs = model(torch.randn(4, 3, 32, 32))

    assert outputs.shape == (4, 2)


def test_frozen_encoder_classifier_supports_tuple_output() -> None:
    model = FrozenEncoderClassifier(
        encoder=FakeTupleEncoder(feature_dim=8),
        feature_dim=8,
        num_classes=2,
    )

    outputs = model(torch.randn(4, 3, 32, 32))

    assert outputs.shape == (4, 2)


def test_frozen_encoder_classifier_rejects_unsupported_output() -> None:
    model = FrozenEncoderClassifier(
        encoder=UnsupportedEncoder(),
        feature_dim=8,
        num_classes=2,
    )

    with pytest.raises(TypeError, match="Unsupported encoder output type"):
        model(torch.randn(4, 3, 32, 32))


def test_build_retfound_linear_classifier_requires_existing_checkpoint(tmp_path: Path) -> None:
    missing_checkpoint = tmp_path / "missing_checkpoint.pth"

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        build_retfound_linear_classifier(
            checkpoint_path=missing_checkpoint,
            num_classes=2,
        )


def test_build_retfound_linear_classifier_is_explicitly_not_implemented(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "retfound_checkpoint.pth"
    checkpoint_path.write_bytes(b"fake checkpoint")

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        build_retfound_linear_classifier(
            checkpoint_path=checkpoint_path,
            num_classes=2,
        )
