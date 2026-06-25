"""Train an SNGP-style cached-feature head on frozen RETFound feature bundles."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from uncertainty_retfound.data.dataloaders import create_dataloader
from uncertainty_retfound.evaluation.metrics import classification_summary
from uncertainty_retfound.models.sngp import SNGPFeatureClassifier
from uncertainty_retfound.training.loops import train_one_epoch

_SELECTION_METRIC_DIRECTIONS: dict[str, str] = {
    "val_loss": "min",
    "accuracy": "max",
    "balanced_accuracy": "max",
    "sensitivity": "max",
}


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


def _fit_precision_from_dataloader(
    model: SNGPFeatureClassifier,
    dataloader: Any,
    device: torch.device,
    show_progress: bool,
    progress_description: str,
) -> None:
    """Recompute the diagonal precision estimate from the training split."""

    model.eval()
    model.reset_precision()

    dataloader_iterator = tqdm(
        dataloader,
        desc=progress_description,
        disable=not show_progress,
        leave=False,
    )

    with torch.no_grad():
        for batch in dataloader_iterator:
            features = batch["image"]
            if not isinstance(features, torch.Tensor):
                raise TypeError("batch['image'] must be a torch.Tensor.")

            random_features = model.random_feature_activations(features.to(device))
            model.update_precision(random_features)


def _evaluate_sngp_model(
    model: SNGPFeatureClassifier,
    dataloader: Any,
    device: torch.device,
    show_progress: bool,
    progress_description: str,
) -> dict[str, Any]:
    """Evaluate an SNGP-style cached-feature head."""

    total_loss = 0.0
    num_batches = 0
    num_examples = 0
    label_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []
    probability_batches: list[torch.Tensor] = []
    logit_batches: list[torch.Tensor] = []
    predictive_entropy_batches: list[torch.Tensor] = []
    sngp_variance_batches: list[torch.Tensor] = []
    sngp_uncertainty_batches: list[torch.Tensor] = []
    image_paths: list[str] = []
    id_codes: list[str] = []

    model.eval()
    dataloader_iterator = tqdm(
        dataloader,
        desc=progress_description,
        disable=not show_progress,
        leave=False,
    )

    with torch.no_grad():
        for batch in dataloader_iterator:
            features = batch["image"]
            labels = batch["label"]

            if not isinstance(features, torch.Tensor):
                raise TypeError("batch['image'] must be a torch.Tensor.")
            if not isinstance(labels, torch.Tensor):
                raise TypeError("batch['label'] must be a torch.Tensor.")

            features = features.to(device)
            labels = labels.to(device)

            predictive_stats = model.predictive_statistics(features)
            logits = predictive_stats["logits"]
            probabilities = predictive_stats["probabilities"]
            predictions = predictive_stats["predictions"]
            predictive_entropy = predictive_stats["predictive_entropy"]
            sngp_variance = predictive_stats["sngp_variance"]
            sngp_uncertainty = predictive_stats["sngp_uncertainty"]

            batch_loss = float(F.cross_entropy(logits, labels).item())
            batch_size = int(labels.shape[0])

            total_loss += batch_loss * batch_size
            num_batches += 1
            num_examples += batch_size

            label_batches.append(labels.detach().cpu())
            prediction_batches.append(predictions.detach().cpu())
            probability_batches.append(probabilities.detach().cpu())
            logit_batches.append(logits.detach().cpu())
            predictive_entropy_batches.append(predictive_entropy.detach().cpu())
            sngp_variance_batches.append(sngp_variance.detach().cpu())
            sngp_uncertainty_batches.append(sngp_uncertainty.detach().cpu())
            id_codes.extend([str(value) for value in batch["id_code"]])
            image_paths.extend([str(value) for value in batch["image_path"]])

            if show_progress:
                dataloader_iterator.set_postfix_str(f"loss={batch_loss:.4f}")

    if num_batches == 0:
        raise ValueError("Cannot evaluate with an empty dataloader.")

    labels = torch.cat(label_batches, dim=0)
    predictions = torch.cat(prediction_batches, dim=0)
    probabilities = torch.cat(probability_batches, dim=0)
    logits = torch.cat(logit_batches, dim=0)
    predictive_entropy = torch.cat(predictive_entropy_batches, dim=0)
    sngp_variance = torch.cat(sngp_variance_batches, dim=0)
    sngp_uncertainty = torch.cat(sngp_uncertainty_batches, dim=0)

    metrics = classification_summary(
        predictions=predictions,
        labels=labels,
        num_classes=int(probabilities.shape[1]),
        positive_scores=probabilities[:, 1] if probabilities.shape[1] == 2 else None,
        probabilities=probabilities,
        logits=logits,
    )

    return {
        "loss": total_loss / num_examples,
        "num_batches": num_batches,
        "num_examples": num_examples,
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "logits": logits,
        "predictive_entropy": predictive_entropy,
        "sngp_variance": sngp_variance,
        "sngp_uncertainty": sngp_uncertainty,
        "image_paths": image_paths,
        "id_codes": id_codes,
        "metrics": metrics,
    }


def _write_validation_predictions_csv(
    validation_result: dict[str, Any],
    output_path: Path,
) -> None:
    """Write SNGP predictive statistics to CSV."""

    probabilities = validation_result.get("probabilities")
    logits = validation_result.get("logits")
    labels = validation_result.get("labels")
    predictions = validation_result.get("predictions")
    predictive_entropy = validation_result.get("predictive_entropy")
    sngp_variance = validation_result.get("sngp_variance")
    sngp_uncertainty = validation_result.get("sngp_uncertainty")
    image_paths = validation_result.get("image_paths")
    id_codes = validation_result.get("id_codes")

    if not isinstance(probabilities, torch.Tensor) or not isinstance(logits, torch.Tensor):
        raise ValueError("Validation result does not contain probability/logit tensors.")
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            "validation_predictions.csv currently supports binary classification outputs only."
        )
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            "validation_predictions.csv currently supports binary classification outputs only."
        )

    required_tensors = (
        labels,
        predictions,
        predictive_entropy,
        sngp_variance,
        sngp_uncertainty,
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in required_tensors):
        raise ValueError("Validation result is missing SNGP predictive tensors.")
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
            "predictive_entropy": predictive_entropy.tolist(),
            "sngp_variance": sngp_variance.tolist(),
            "sngp_uncertainty": sngp_uncertainty.tolist(),
        }
    )
    dataframe.to_csv(output_path, index=False)


def _extract_selection_metric_value(
    epoch_result: dict[str, Any],
    selection_metric: str,
) -> float:
    """Extract the scalar value used for best-epoch selection."""

    if selection_metric == "val_loss":
        metric_value = epoch_result.get("val_loss")
    else:
        val_metrics = epoch_result.get("val_metrics")
        if not isinstance(val_metrics, dict):
            raise ValueError(
                f"Epoch result is missing 'val_metrics' for selection metric '{selection_metric}'."
            )
        metric_value = val_metrics.get(selection_metric)

    if not isinstance(metric_value, (int, float)):
        raise ValueError(
            f"Selection metric '{selection_metric}' is missing or non-scalar in epoch results."
        )

    return float(metric_value)


def _is_better_epoch(
    current_value: float,
    best_value: float | None,
    selection_metric: str,
) -> bool:
    """Return True when the current metric improves on the best so far."""

    if best_value is None:
        return True

    direction = _SELECTION_METRIC_DIRECTIONS[selection_metric]
    if direction == "min":
        return current_value < best_value

    return current_value > best_value


def run_sngp_feature_head_training(
    train_features: str | Path,
    val_features: str | Path,
    output_dir: str | Path,
    num_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str | None = "cpu",
    hidden_dim: int = 0,
    rff_dim: int = 1024,
    kernel_scale: float = 1.0,
    ridge_penalty: float = 1.0,
    selection_metric: str = "val_loss",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Train an SNGP-style cached-feature classifier."""

    if epochs <= 0:
        raise ValueError(f"epochs must be positive. Got: {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive. Got: {batch_size}")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive. Got: {learning_rate}")
    if selection_metric not in _SELECTION_METRIC_DIRECTIONS:
        raise ValueError(
            "selection_metric must be one of: "
            f"{sorted(_SELECTION_METRIC_DIRECTIONS)}. Got: {selection_metric}"
        )

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
    precision_dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
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

    model = SNGPFeatureClassifier(
        input_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        rff_dim=rff_dim,
        kernel_scale=kernel_scale,
        ridge_penalty=ridge_penalty,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_results: list[dict[str, Any]] = []
    final_validation_result: dict[str, Any] | None = None
    best_validation_result: dict[str, Any] | None = None
    best_epoch_result: dict[str, Any] | None = None
    best_metric_value: float | None = None
    best_model_state_dict: dict[str, torch.Tensor] | None = None

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
        _fit_precision_from_dataloader(
            model=model,
            dataloader=precision_dataloader,
            device=resolved_device,
            show_progress=show_progress,
            progress_description=f"Fit precision epoch {epoch_number}/{epochs}",
        )
        validation_result = _evaluate_sngp_model(
            model=model,
            dataloader=val_dataloader,
            device=resolved_device,
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

        current_metric_value = _extract_selection_metric_value(epoch_result, selection_metric)
        if _is_better_epoch(current_metric_value, best_metric_value, selection_metric):
            best_metric_value = current_metric_value
            best_epoch_result = epoch_result
            best_validation_result = {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in validation_result.items()
            }
            best_model_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch_number}/{epochs} "
            f"train_loss={epoch_result['train_loss']:.4f} "
            f"val_loss={epoch_result['val_loss']:.4f} "
            f"val_accuracy={epoch_result['val_accuracy']:.4f}"
        )

    if final_validation_result is None:
        raise RuntimeError("Validation result was not produced during training.")
    if best_validation_result is None or best_epoch_result is None or best_model_state_dict is None:
        raise RuntimeError("Best epoch result was not produced during training.")

    validation_predictions_path = output_dir_path / "validation_predictions.csv"
    _write_validation_predictions_csv(
        validation_result=final_validation_result,
        output_path=validation_predictions_path,
    )
    best_validation_predictions_path = output_dir_path / "best_validation_predictions.csv"
    _write_validation_predictions_csv(
        validation_result=best_validation_result,
        output_path=best_validation_predictions_path,
    )

    model_path = output_dir_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            **model.to_serializable_config(),
        },
        model_path,
    )

    best_model_path = output_dir_path / "best_model.pt"
    torch.save(
        {
            "model_state_dict": best_model_state_dict,
            **model.to_serializable_config(),
        },
        best_model_path,
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
        "hidden_dim": int(hidden_dim),
        "rff_dim": int(rff_dim),
        "kernel_scale": float(kernel_scale),
        "ridge_penalty": float(ridge_penalty),
        "best_epoch": int(best_epoch_result["epoch"]),
        "best_metric_name": selection_metric,
        "best_metric_value": float(best_metric_value),
    }
    config_path = output_dir_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_metrics = {
        **config,
        "train_split": str(train_bundle["split"]),
        "validation_split": str(val_bundle["split"]),
        "epoch_result": best_epoch_result,
        "final_validation_metrics": best_epoch_result["val_metrics"],
        "validation_predictions_path": str(best_validation_predictions_path),
        "model_path": str(best_model_path),
    }
    best_metrics_path = output_dir_path / "best_metrics.json"
    best_metrics_path.write_text(
        json.dumps(_serialize_for_json(best_metrics), indent=2),
        encoding="utf-8",
    )

    metrics = {
        **config,
        "train_split": str(train_bundle["split"]),
        "validation_split": str(val_bundle["split"]),
        "epoch_results": epoch_results,
        "final_validation_metrics": epoch_results[-1]["val_metrics"],
        "validation_predictions_path": str(validation_predictions_path),
        "model_path": str(model_path),
        "best_validation_predictions_path": str(best_validation_predictions_path),
        "best_model_path": str(best_model_path),
        "best_metrics_path": str(best_metrics_path),
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
        "best_metrics": best_metrics,
    }


def parse_args() -> argparse.Namespace:
    """Parse cached-feature SNGP-style head training CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Train an SNGP-style cached-feature head on frozen RETFound features."
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
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--rff-dim", type=int, default=1024)
    parser.add_argument("--kernel-scale", type=float, default=1.0)
    parser.add_argument("--ridge-penalty", type=float, default=1.0)
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=tuple(_SELECTION_METRIC_DIRECTIONS),
        default="val_loss",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the cached-feature SNGP-style training CLI."""

    args = parse_args()
    result = run_sngp_feature_head_training(
        train_features=args.train_features,
        val_features=args.val_features,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        hidden_dim=args.hidden_dim,
        rff_dim=args.rff_dim,
        kernel_scale=args.kernel_scale,
        ridge_penalty=args.ridge_penalty,
        selection_metric=args.selection_metric,
        show_progress=not args.no_progress,
    )
    print(f"Saved metrics to: {result['metrics_path']}")
    print(f"Saved best metrics to: {result['best_metrics_path']}")


if __name__ == "__main__":
    main()
