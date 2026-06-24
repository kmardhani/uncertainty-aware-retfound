"""Train a linear softmax head on cached RETFound features."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from uncertainty_retfound.data.dataloaders import create_dataloader
from uncertainty_retfound.evaluation.loops import evaluate_model
from uncertainty_retfound.training.loops import train_one_epoch


def _set_seed(seed: int) -> None:
    """Set random seeds for deterministic local smoke tests."""

    random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(device: str | None) -> torch.device:
    """Resolve a requested device into a concrete torch device."""

    if device in (None, "cpu"):
        return torch.device("cpu")

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def _serialize_for_json(value: Any) -> Any:
    """Convert nested experiment results into JSON-serializable values."""

    if isinstance(value, torch.Tensor):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {key: _serialize_for_json(nested_value) for key, nested_value in value.items()}

    if isinstance(value, list):
        return [_serialize_for_json(item) for item in value]

    return value


def _write_validation_predictions_csv(
    validation_result: dict[str, Any],
    output_path: Path,
) -> None:
    """Write per-example validation predictions to CSV."""

    logits = validation_result.get("logits")
    probabilities = validation_result.get("probabilities")
    if not isinstance(logits, torch.Tensor):
        raise ValueError("Validation result does not contain logit tensors.")

    if not isinstance(probabilities, torch.Tensor):
        raise ValueError("Validation result does not contain probability tensors.")

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            "validation_predictions.csv currently supports binary classification outputs only."
        )

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            "validation_predictions.csv currently supports binary classification outputs only."
        )

    labels = validation_result["labels"]
    predictions = validation_result["predictions"]
    image_paths = validation_result.get("image_paths")
    id_codes = validation_result.get("id_codes")

    if not isinstance(labels, torch.Tensor) or not isinstance(predictions, torch.Tensor):
        raise ValueError("Validation result is missing label or prediction tensors.")

    if not isinstance(image_paths, list) or not isinstance(id_codes, list):
        raise ValueError("Validation result is missing image path or id_code values.")

    confidence = torch.max(probabilities, dim=1).values
    dataframe = pd.DataFrame(
        {
            "id_code": id_codes,
            "image_path": image_paths,
            "true_label": labels.tolist(),
            "predicted_label": predictions.tolist(),
            "logit_class_0": logits[:, 0].tolist(),
            "logit_class_1": logits[:, 1].tolist(),
            "probability_class_0": probabilities[:, 0].tolist(),
            "probability_class_1": probabilities[:, 1].tolist(),
            "confidence": confidence.tolist(),
            "is_correct": (predictions == labels).tolist(),
        }
    )
    dataframe.to_csv(output_path, index=False)


def _load_feature_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Load and validate a cached feature bundle."""

    resolved_path = Path(bundle_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Feature bundle not found: {resolved_path}")

    bundle = torch.load(resolved_path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise ValueError(f"Feature bundle must contain a dictionary: {resolved_path}")

    required_keys = {"features", "labels", "id_codes", "image_paths", "split"}
    missing_keys = required_keys - set(bundle.keys())
    if missing_keys:
        raise ValueError(
            f"Feature bundle is missing required keys {sorted(missing_keys)}: {resolved_path}"
        )

    features = bundle["features"]
    labels = bundle["labels"]
    id_codes = bundle["id_codes"]
    image_paths = bundle["image_paths"]

    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError(f"Feature bundle 'features' must be a 2D tensor: {resolved_path}")

    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError(f"Feature bundle 'labels' must be a 1D tensor: {resolved_path}")

    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"Features and labels must have matching row counts: {resolved_path}")

    if not isinstance(id_codes, list) or not isinstance(image_paths, list):
        raise ValueError(f"Feature bundle id_codes/image_paths must be lists: {resolved_path}")

    if len(id_codes) != features.shape[0] or len(image_paths) != features.shape[0]:
        raise ValueError(
            f"Feature bundle metadata lengths must match feature rows: {resolved_path}"
        )

    return {
        "features": features.to(dtype=torch.float32),
        "labels": labels.to(dtype=torch.int64),
        "id_codes": [str(value) for value in id_codes],
        "image_paths": [str(value) for value in image_paths],
        "split": str(bundle["split"]),
    }


class CachedFeatureDataset(Dataset[dict[str, Any]]):
    """Dataset wrapper over cached feature bundles."""

    def __init__(self, feature_bundle: dict[str, Any]) -> None:
        self.features = feature_bundle["features"]
        self.labels = feature_bundle["labels"]
        self.id_codes = feature_bundle["id_codes"]
        self.image_paths = feature_bundle["image_paths"]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.features[index],
            "label": self.labels[index],
            "id_code": self.id_codes[index],
            "image_path": self.image_paths[index],
        }


def run_feature_head_training(
    train_features: str | Path,
    val_features: str | Path,
    output_dir: str | Path,
    num_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str | None = "cpu",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Train a linear classifier on cached feature bundles."""

    if epochs <= 0:
        raise ValueError(f"epochs must be positive. Got: {epochs}")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive. Got: {batch_size}")

    _set_seed(seed)

    train_features_path = Path(train_features)
    val_features_path = Path(val_features)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    train_bundle = _load_feature_bundle(train_features_path)
    val_bundle = _load_feature_bundle(val_features_path)

    feature_dim = int(train_bundle["features"].shape[1])
    if int(val_bundle["features"].shape[1]) != feature_dim:
        raise ValueError(
            "Train and validation feature bundles must have matching feature dimensions."
        )

    train_dataset = CachedFeatureDataset(train_bundle)
    val_dataset = CachedFeatureDataset(val_bundle)

    resolved_device = _resolve_device(device)
    pin_memory = resolved_device.type != "cpu"

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = nn.Linear(feature_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_results: list[dict[str, Any]] = []
    final_validation_result: dict[str, Any] | None = None

    for epoch_index in range(epochs):
        epoch_number = epoch_index + 1
        train_result = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=resolved_device,
            show_progress=show_progress,
            progress_description=f"Train epoch {epoch_number}/{epochs}",
        )
        validation_result = evaluate_model(
            model=model,
            dataloader=val_dataloader,
            device=resolved_device,
            include_metrics=True,
            show_progress=show_progress,
            progress_description=f"Val epoch {epoch_number}/{epochs}",
        )
        final_validation_result = validation_result

        validation_metrics = validation_result["metrics"]
        epoch_result = {
            "epoch": epoch_number,
            "train_loss": float(train_result["loss"]),
            "val_loss": float(validation_result["loss"]),
            "val_accuracy": float(validation_metrics["accuracy"]),
            "val_metrics": validation_metrics,
        }
        epoch_results.append(epoch_result)

        print(
            f"Epoch {epoch_number}/{epochs} "
            f"train_loss={epoch_result['train_loss']:.4f} "
            f"val_loss={epoch_result['val_loss']:.4f} "
            f"val_accuracy={epoch_result['val_accuracy']:.4f}"
        )

    if final_validation_result is None:
        raise RuntimeError("Validation result was not produced during training.")

    validation_predictions_path = output_dir_path / "validation_predictions.csv"
    _write_validation_predictions_csv(
        validation_result=final_validation_result,
        output_path=validation_predictions_path,
    )

    model_path = output_dir_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_dim": feature_dim,
            "num_classes": int(num_classes),
        },
        model_path,
    )

    config = {
        "train_features": str(train_features_path),
        "val_features": str(val_features_path),
        "output_dir": str(output_dir_path),
        "num_classes": int(num_classes),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "device": str(resolved_device),
        "feature_dim": feature_dim,
    }
    config_path = output_dir_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics = {
        **config,
        "train_split": str(train_bundle["split"]),
        "validation_split": str(val_bundle["split"]),
        "epoch_results": epoch_results,
        "final_validation_metrics": epoch_results[-1]["val_metrics"],
        "validation_predictions_path": str(validation_predictions_path),
        "model_path": str(model_path),
    }
    metrics_path = output_dir_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(_serialize_for_json(metrics), indent=2),
        encoding="utf-8",
    )

    return {
        **metrics,
        "metrics_path": str(metrics_path),
        "config_path": str(config_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse cached-feature head training CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Train a linear softmax head on cached RETFound features."
    )
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the cached-feature head training CLI."""

    args = parse_args()
    result = run_feature_head_training(
        train_features=args.train_features,
        val_features=args.val_features,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        show_progress=not args.no_progress,
    )
    print(f"Saved metrics to: {result['metrics_path']}")


if __name__ == "__main__":
    main()
