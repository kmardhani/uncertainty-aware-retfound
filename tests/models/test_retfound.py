"""Tests for RETFound-style wrapper models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
from torch import nn

from uncertainty_retfound.models.retfound import (
    FrozenEncoderClassifier,
    build_retfound_linear_classifier,
    load_external_retfound_encoder,
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


def _write_fake_models_vit(repo_path: Path) -> None:
    repo_path.mkdir(parents=True, exist_ok=True)
    (repo_path / "models_vit.py").write_text(
        """
import torch
from torch import nn


class FakeRETFoundEncoder(nn.Module):
    def __init__(self, feature_dim: int = 8, num_classes: int = 1000) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(3, feature_dim)
        self.head = nn.Linear(feature_dim, 1000)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(inputs).flatten(1)
        return self.proj(pooled)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        return self.head(features)


def RETFound_mae(**kwargs) -> nn.Module:
    return FakeRETFoundEncoder(feature_dim=8, **kwargs)
""".strip(),
        encoding="utf-8",
    )


def _build_fake_checkpoint(
    repo_path: Path,
    checkpoint_path: Path,
    *,
    container_key: str = "model",
    prefix: str = "",
    include_mismatched_head: bool = False,
    include_args_namespace: bool = False,
) -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("fake_models_vit_for_test", repo_path / "models_vit.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load fake models_vit.py for test setup.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["fake_models_vit_for_test"] = module
    spec.loader.exec_module(module)

    model = module.RETFound_mae()
    state_dict = model.state_dict()
    checkpoint_state_dict: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        checkpoint_state_dict[f"{prefix}{key}"] = value.clone()

    if include_mismatched_head:
        checkpoint_state_dict[f"{prefix}head.weight"] = torch.randn(5, 8)
        checkpoint_state_dict[f"{prefix}head.bias"] = torch.randn(5)

    checkpoint_payload: dict[str, object] = {container_key: checkpoint_state_dict}

    if include_args_namespace:
        checkpoint_payload["args"] = argparse.Namespace(model="RETFound_mae", img_size=224)

    torch.save(checkpoint_payload, checkpoint_path)


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


def test_load_external_retfound_encoder_requires_existing_repo_path(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"fake checkpoint")
    missing_repo = tmp_path / "missing_repo"

    with pytest.raises(FileNotFoundError, match="repo path not found"):
        load_external_retfound_encoder(
            retfound_repo_path=missing_repo,
            checkpoint_path=checkpoint_path,
        )


def test_load_external_retfound_encoder_requires_existing_checkpoint(tmp_path: Path) -> None:
    repo_path = tmp_path / "fake_retfound_repo"
    _write_fake_models_vit(repo_path)
    missing_checkpoint = tmp_path / "missing_checkpoint.pth"

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_external_retfound_encoder(
            retfound_repo_path=repo_path,
            checkpoint_path=missing_checkpoint,
        )


def test_load_external_retfound_encoder_rejects_missing_models_vit_import(
    tmp_path: Path,
) -> None:
    repo_path = tmp_path / "fake_retfound_repo"
    repo_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"fake checkpoint")

    with pytest.raises(ImportError, match="Could not import 'models_vit'"):
        load_external_retfound_encoder(
            retfound_repo_path=repo_path,
            checkpoint_path=checkpoint_path,
        )


def test_load_external_retfound_encoder_rejects_missing_architecture(tmp_path: Path) -> None:
    repo_path = tmp_path / "fake_retfound_repo"
    _write_fake_models_vit(repo_path)
    checkpoint_path = tmp_path / "checkpoint.pth"
    _build_fake_checkpoint(repo_path, checkpoint_path)

    with pytest.raises(AttributeError, match="Architecture 'missing_arch'"):
        load_external_retfound_encoder(
            retfound_repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            architecture="missing_arch",
        )


def test_load_external_retfound_encoder_loads_fake_external_repo_checkpoint(
    tmp_path: Path,
) -> None:
    repo_path = tmp_path / "fake_retfound_repo"
    checkpoint_path = tmp_path / "checkpoint.pth"
    _write_fake_models_vit(repo_path)
    _build_fake_checkpoint(repo_path, checkpoint_path)

    encoder = load_external_retfound_encoder(
        retfound_repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        feature_dim=8,
    )
    model = FrozenEncoderClassifier(
        encoder=encoder,
        feature_dim=8,
        num_classes=2,
    )

    outputs = model(torch.randn(4, 3, 32, 32))

    assert outputs.shape == (4, 2)


def test_load_external_retfound_encoder_supports_module_prefix_stripping(
    tmp_path: Path,
) -> None:
    repo_path = tmp_path / "fake_retfound_repo"
    checkpoint_path = tmp_path / "checkpoint.pth"
    _write_fake_models_vit(repo_path)
    _build_fake_checkpoint(repo_path, checkpoint_path, prefix="module.")

    encoder = load_external_retfound_encoder(
        retfound_repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        feature_dim=8,
    )

    outputs = encoder(torch.randn(2, 3, 32, 32))
    assert outputs.shape == (2, 8)


def test_load_external_retfound_encoder_ignores_mismatched_head_keys(
    tmp_path: Path,
) -> None:
    repo_path = tmp_path / "fake_retfound_repo"
    checkpoint_path = tmp_path / "checkpoint.pth"
    _write_fake_models_vit(repo_path)
    _build_fake_checkpoint(
        repo_path,
        checkpoint_path,
        prefix="module.",
        include_mismatched_head=True,
    )

    encoder = load_external_retfound_encoder(
        retfound_repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        feature_dim=8,
    )

    outputs = encoder(torch.randn(2, 3, 32, 32))
    assert outputs.shape == (2, 8)


def test_load_external_retfound_encoder_supports_checkpoint_with_argparse_namespace(
    tmp_path: Path,
) -> None:
    repo_path = tmp_path / "fake_retfound_repo"
    checkpoint_path = tmp_path / "checkpoint_with_args.pth"
    _write_fake_models_vit(repo_path)
    _build_fake_checkpoint(
        repo_path,
        checkpoint_path,
        include_args_namespace=True,
    )

    encoder = load_external_retfound_encoder(
        retfound_repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        feature_dim=8,
    )
    model = FrozenEncoderClassifier(
        encoder=encoder,
        feature_dim=8,
        num_classes=2,
    )

    outputs = model(torch.randn(2, 3, 32, 32))
    assert outputs.shape == (2, 2)


def test_build_retfound_linear_classifier_requires_repo_path(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"fake checkpoint")

    with pytest.raises(ValueError, match="retfound_repo_path is required"):
        build_retfound_linear_classifier(
            checkpoint_path=checkpoint_path,
            num_classes=2,
        )


def test_build_retfound_linear_classifier_uses_external_repo_adapter(tmp_path: Path) -> None:
    repo_path = tmp_path / "fake_retfound_repo"
    checkpoint_path = tmp_path / "checkpoint.pth"
    _write_fake_models_vit(repo_path)
    _build_fake_checkpoint(repo_path, checkpoint_path)

    model = build_retfound_linear_classifier(
        checkpoint_path=checkpoint_path,
        num_classes=2,
        retfound_repo_path=repo_path,
        feature_dim=8,
    )

    outputs = model(torch.randn(4, 3, 32, 32))

    assert outputs.shape == (4, 2)
