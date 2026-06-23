"""RETFound-style encoder wrappers and staged checkpoint-loading boundaries."""

from __future__ import annotations

import argparse
import importlib
import inspect
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn


def _extract_features_from_encoder_output(
    encoder_output: Any,
    feature_dim: int,
) -> torch.Tensor:
    """Normalize common encoder outputs into a 2D feature tensor."""

    tensor_output: torch.Tensor | None = None

    if isinstance(encoder_output, torch.Tensor):
        tensor_output = encoder_output
    elif isinstance(encoder_output, dict):
        if "features" in encoder_output:
            features_value = encoder_output["features"]
            if not isinstance(features_value, torch.Tensor):
                raise TypeError("Encoder output key 'features' must contain a torch.Tensor.")
            tensor_output = features_value
        elif "pooler_output" in encoder_output:
            pooler_output = encoder_output["pooler_output"]
            if not isinstance(pooler_output, torch.Tensor):
                raise TypeError("Encoder output key 'pooler_output' must contain a torch.Tensor.")
            tensor_output = pooler_output
        elif "last_hidden_state" in encoder_output:
            hidden_state = encoder_output["last_hidden_state"]
            if not isinstance(hidden_state, torch.Tensor):
                raise TypeError(
                    "Encoder output key 'last_hidden_state' must contain a torch.Tensor."
                )
            tensor_output = hidden_state
        else:
            raise ValueError(
                "Unsupported encoder output dictionary. Expected one of: "
                "'features', 'pooler_output', 'last_hidden_state'."
            )
    elif isinstance(encoder_output, (tuple, list)):
        if not encoder_output:
            raise ValueError("Encoder output tuple/list must not be empty.")
        return _extract_features_from_encoder_output(encoder_output[0], feature_dim)
    else:
        raise TypeError(
            "Unsupported encoder output type. Expected torch.Tensor, dict, tuple, or list."
        )

    if tensor_output.ndim == 2:
        features = tensor_output
    elif tensor_output.ndim == 3:
        features = tensor_output[:, 0, :]
    else:
        raise ValueError(
            "Unsupported encoder tensor shape. Expected [batch, dim] or [batch, tokens, dim]."
        )

    if features.shape[1] != feature_dim:
        raise ValueError(
            f"Encoder feature dimension mismatch. Expected {feature_dim}, "
            f"got {int(features.shape[1])}."
        )

    return features


class FrozenEncoderClassifier(nn.Module):
    """A frozen-encoder plus linear-head classifier."""

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        num_classes: int,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.head = nn.Linear(feature_dim, num_classes)

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(inputs)
        features = _extract_features_from_encoder_output(
            encoder_output=encoder_output,
            feature_dim=self.feature_dim,
        )
        return self.head(features)


class ForwardFeaturesEncoder(nn.Module):
    """Wrap an encoder and expose ``forward_features`` as the forward path."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, inputs: torch.Tensor) -> Any:
        forward_features = getattr(self.encoder, "forward_features", None)
        if not callable(forward_features):
            raise AttributeError("Wrapped encoder does not define callable forward_features().")
        return forward_features(inputs)


@contextmanager
def _temporary_sys_path(path: Path) -> Any:
    """Temporarily prepend a path to sys.path for external repo imports."""

    path_string = str(path)
    sys.path.insert(0, path_string)
    try:
        yield
    finally:
        if path_string in sys.path:
            sys.path.remove(path_string)


def _import_models_vit_module(retfound_repo_path: Path) -> ModuleType:
    """Import the external RETFound models_vit module from a local repo path."""

    previous_module = sys.modules.pop("models_vit", None)

    try:
        with _temporary_sys_path(retfound_repo_path):
            try:
                return importlib.import_module("models_vit")
            except ImportError as error:
                raise ImportError(
                    "Could not import 'models_vit' from the external RETFound repo path: "
                    f"{retfound_repo_path}"
                ) from error
    finally:
        sys.modules.pop("models_vit", None)
        if previous_module is not None:
            sys.modules["models_vit"] = previous_module


def _instantiate_external_architecture(architecture_builder: Any) -> nn.Module:
    """Instantiate an external RETFound architecture with num_classes=0 when supported."""

    signature = inspect.signature(architecture_builder)
    parameters = signature.parameters
    supports_num_classes = "num_classes" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )

    if supports_num_classes:
        model = architecture_builder(num_classes=0)
    else:
        model = architecture_builder()

    if not isinstance(model, nn.Module):
        raise TypeError("External RETFound architecture did not return a torch.nn.Module.")

    return model


def _extract_checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a state dict from common checkpoint container formats."""

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "teacher" in checkpoint:
            state_dict = checkpoint["teacher"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError("Unsupported checkpoint format. Expected a state-dict-like dictionary.")

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint did not contain a valid state dictionary.")

    normalized_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_state_dict[str(key)] = value

    return normalized_state_dict


def _load_checkpoint_file(checkpoint: Path) -> object:
    """Load a checkpoint file with explicit PyTorch safe-loading behavior.

    For PyTorch versions that support ``weights_only``, this first uses
    ``weights_only=True``. If the checkpoint contains a trusted
    ``argparse.Namespace`` metadata object, it retries with a narrow allowlist.
    As a final explicit fallback for trusted external research checkpoints only,
    it uses ``weights_only=False``.
    """

    torch_load_parameters = inspect.signature(torch.load).parameters
    supports_weights_only = "weights_only" in torch_load_parameters

    if not supports_weights_only:
        try:
            return torch.load(checkpoint, map_location="cpu")
        except Exception as error:
            raise RuntimeError(f"Could not load RETFound checkpoint: {checkpoint}") from error

    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        if "argparse.Namespace" in str(error):
            try:
                with torch.serialization.safe_globals([argparse.Namespace]):
                    return torch.load(checkpoint, map_location="cpu", weights_only=True)
            except Exception:
                pass

    try:
        # Final explicit fallback for trusted external research checkpoints only.
        # Some third-party checkpoints may contain small non-tensor metadata that
        # still prevents weights_only loading even after a narrow allowlist retry.
        return torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as fallback_error:
        raise RuntimeError(f"Could not load RETFound checkpoint: {checkpoint}") from fallback_error


def _strip_common_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip common distributed/backbone prefixes from checkpoint keys."""

    normalized_state_dict: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        normalized_key = key
        for prefix in ("module.", "backbone."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
        normalized_state_dict[normalized_key] = value

    return normalized_state_dict


def _remove_mismatched_head_keys(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Remove common classification-head keys when shapes do not match."""

    model_state_dict = model.state_dict()
    filtered_state_dict: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        model_value = model_state_dict.get(key)
        is_head_key = key.endswith("head.weight") or key.endswith("head.bias")

        if is_head_key and model_value is not None and model_value.shape != value.shape:
            continue

        filtered_state_dict[key] = value

    return filtered_state_dict


def load_external_retfound_encoder(
    retfound_repo_path: str | Path,
    checkpoint_path: str | Path,
    architecture: str = "RETFound_mae",
    feature_dim: int = 1024,
) -> torch.nn.Module:
    """Load a RETFound-style encoder from a separate local external repo.

    This function does not download code or weights. It expects a local clone of
    the external RETFound repo and a local checkpoint path.
    """

    repo_path = Path(retfound_repo_path)
    checkpoint = Path(checkpoint_path)

    if not repo_path.exists():
        raise FileNotFoundError(f"RETFound repo path not found: {repo_path}")
    if not repo_path.is_dir():
        raise ValueError(f"RETFound repo path is not a directory: {repo_path}")

    if not checkpoint.exists():
        raise FileNotFoundError(f"RETFound checkpoint not found: {checkpoint}")
    if not checkpoint.is_file():
        raise ValueError(f"RETFound checkpoint path is not a file: {checkpoint}")

    models_vit = _import_models_vit_module(repo_path)
    architecture_builder = getattr(models_vit, architecture, None)

    if architecture_builder is None or not callable(architecture_builder):
        raise AttributeError(
            f"Architecture '{architecture}' was not found in external module "
            f"'models_vit' from {repo_path}."
        )

    encoder = _instantiate_external_architecture(architecture_builder)

    checkpoint_data = _load_checkpoint_file(checkpoint)

    state_dict = _extract_checkpoint_state_dict(checkpoint_data)
    state_dict = _strip_common_prefixes(state_dict)
    state_dict = _remove_mismatched_head_keys(state_dict, encoder)

    try:
        encoder.load_state_dict(state_dict, strict=False)
    except Exception as error:
        raise RuntimeError(
            "Failed to load checkpoint state into the external RETFound encoder. "
            f"Checkpoint: {checkpoint}"
        ) from error

    if callable(getattr(encoder, "forward_features", None)):
        return ForwardFeaturesEncoder(encoder)

    return encoder


def build_retfound_linear_classifier(
    checkpoint_path: str | Path,
    num_classes: int,
    retfound_repo_path: str | Path | None = None,
    feature_dim: int = 1024,
    freeze_encoder: bool = True,
    architecture: str = "RETFound_mae",
) -> FrozenEncoderClassifier:
    """Build a frozen linear-head classifier on top of an external RETFound encoder."""

    if retfound_repo_path is None:
        raise ValueError(
            "retfound_repo_path is required for RETFound loading. "
            "Provide a local external RETFound repo path; this code does not "
            "vendor or auto-download RETFound."
        )

    encoder = load_external_retfound_encoder(
        retfound_repo_path=retfound_repo_path,
        checkpoint_path=checkpoint_path,
        architecture=architecture,
        feature_dim=feature_dim,
    )

    return FrozenEncoderClassifier(
        encoder=encoder,
        feature_dim=feature_dim,
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
    )
