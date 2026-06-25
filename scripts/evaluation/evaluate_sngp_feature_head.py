"""Evaluate a saved SNGP-style cached-feature head on a cached feature bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from uncertainty_retfound.data.dataloaders import create_dataloader
from uncertainty_retfound.models.sngp import SNGPFeatureClassifier
from scripts.training.train_sngp_feature_head import (
    CachedFeatureDataset,
    _evaluate_sngp_model,
    _load_feature_bundle,
    _resolve_device,
    _serialize_for_json,
    _write_validation_predictions_csv,
)


def _load_config(config_json: str | Path) -> dict[str, Any]:
    """Load and validate an SNGP config JSON file."""

    config_path = Path(config_json)
    if not config_path.exists():
        raise FileNotFoundError(f"SNGP config JSON not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"SNGP config JSON must contain an object: {config_path}")

    required_keys = {
        "feature_dim",
        "num_classes",
        "hidden_dim",
        "rff_dim",
        "kernel_scale",
        "ridge_penalty",
    }
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(
            f"SNGP config JSON is missing required keys {sorted(missing_keys)}: {config_path}"
        )

    return config


def _load_model_checkpoint(model_path: str | Path) -> dict[str, Any]:
    """Load and validate a saved SNGP model checkpoint."""

    resolved_model_path = Path(model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"SNGP model checkpoint not found: {resolved_model_path}")

    loaded = torch.load(resolved_model_path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError(f"SNGP model checkpoint must contain a dictionary: {resolved_model_path}")

    if "model_state_dict" not in loaded:
        raise ValueError(
            f"SNGP model checkpoint is missing 'model_state_dict': {resolved_model_path}"
        )

    return loaded


def run_sngp_feature_head_evaluation(
    features: str | Path,
    model_path: str | Path,
    config_json: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    device: str | None = "cpu",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Evaluate a saved SNGP-style cached-feature head without retraining."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive. Got: {batch_size}")

    features_path = Path(features)
    model_checkpoint_path = Path(model_path)
    config_path = Path(config_json)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    config = _load_config(config_path)
    checkpoint = _load_model_checkpoint(model_checkpoint_path)
    feature_bundle = _load_feature_bundle(features_path)

    feature_dim = int(config["feature_dim"])
    if int(feature_bundle["features"].shape[1]) != feature_dim:
        raise ValueError(
            "Feature bundle dimension does not match the saved SNGP config. "
            f"Expected {feature_dim}, got {int(feature_bundle['features'].shape[1])}."
        )

    dataset = CachedFeatureDataset(feature_bundle)
    resolved_device = _resolve_device(device)
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=resolved_device.type != "cpu",
    )

    model = SNGPFeatureClassifier(
        input_dim=feature_dim,
        num_classes=int(config["num_classes"]),
        hidden_dim=int(config["hidden_dim"]),
        rff_dim=int(config["rff_dim"]),
        kernel_scale=float(config["kernel_scale"]),
        ridge_penalty=float(config["ridge_penalty"]),
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluation_result = _evaluate_sngp_model(
        model=model,
        dataloader=dataloader,
        device=resolved_device,
        show_progress=show_progress,
        progress_description=f"Eval {feature_bundle['split']} split",
    )

    predictions_path = output_dir_path / "predictions.csv"
    _write_validation_predictions_csv(
        validation_result=evaluation_result,
        output_path=predictions_path,
    )

    precision_diag = model.precision_diag.detach().cpu()
    metrics = {
        "features": str(features_path),
        "model_path": str(model_checkpoint_path),
        "config_json": str(config_path),
        "output_dir": str(output_dir_path),
        "batch_size": int(batch_size),
        "device": str(resolved_device),
        "evaluated_split": str(feature_bundle["split"]),
        "num_examples": int(evaluation_result["num_examples"]),
        "metrics": evaluation_result["metrics"],
        "loss": float(evaluation_result["loss"]),
        "predictions_path": str(predictions_path),
        "precision_diag_min": float(precision_diag.min().item()),
        "precision_diag_max": float(precision_diag.max().item()),
        "precision_diag_mean": float(precision_diag.mean().item()),
    }
    metrics_path = output_dir_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(_serialize_for_json(metrics), indent=2),
        encoding="utf-8",
    )

    return {
        **metrics,
        "metrics_path": str(metrics_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse evaluation-only SNGP CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate a saved SNGP-style cached-feature head on a cached feature split."
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the SNGP cached-feature evaluation CLI."""

    args = parse_args(argv)
    result = run_sngp_feature_head_evaluation(
        features=args.features,
        model_path=args.model_path,
        config_json=args.config_json,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        show_progress=not args.no_progress,
    )
    print(f"Saved metrics to: {result['metrics_path']}")
    print(f"Saved predictions to: {result['predictions_path']}")


if __name__ == "__main__":
    main()
