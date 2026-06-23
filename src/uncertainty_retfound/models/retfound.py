"""RETFound-style encoder wrappers and staged checkpoint-loading boundaries."""

from __future__ import annotations

from pathlib import Path
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


def build_retfound_linear_classifier(
    checkpoint_path: str | Path,
    num_classes: int,
    feature_dim: int = 1024,
    freeze_encoder: bool = True,
    architecture: str = "vit_large_patch16",
) -> FrozenEncoderClassifier:
    """Validate the checkpoint path and expose a staged RETFound loading boundary.

    Actual RETFound checkpoint loading is intentionally not implemented yet.
    The project needs a compatible ViT/MAE architecture implementation before a
    local checkpoint can be loaded robustly.
    """

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"RETFound checkpoint not found: {checkpoint}")

    raise NotImplementedError(
        "RETFound checkpoint loading is not implemented yet. "
        "This staged API validates the local checkpoint path but still requires "
        "adding a compatible RETFound/MAE ViT architecture loader before a "
        f"checkpoint like '{checkpoint}' can be used for architecture '{architecture}'. "
        "This function intentionally does not auto-download weights."
    )
