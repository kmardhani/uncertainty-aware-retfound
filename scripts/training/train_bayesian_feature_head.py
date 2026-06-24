"""Train a variational Bayesian linear head on cached RETFound features."""

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
from uncertainty_retfound.models.bayesian import BayesianLinearClassifier


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


def _safe_mean_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """Return entropy along the class dimension."""

    clamped_probabilities = torch.clamp(probabilities, min=1e-12, max=1.0)
    return -(clamped_probabilities * torch.log(clamped_probabilities)).sum(dim=-1)


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


def _predictive_statistics(
    model: BayesianLinearClassifier,
    features: torch.Tensor,
    mc_samples: int,
) -> dict[str, torch.Tensor]:
    """Compute Monte Carlo predictive statistics for one batch."""

    if mc_samples <= 0:
        raise ValueError(f"mc_samples must be positive. Got: {mc_samples}")

    logits_samples = [model(features, sample=True) for _ in range(mc_samples)]
    logits_stack = torch.stack(logits_samples, dim=0)
    probability_stack = torch.softmax(logits_stack, dim=-1)
    mean_probabilities = probability_stack.mean(dim=0)
    mean_logits = logits_stack.mean(dim=0)
    predictions = torch.argmax(mean_probabilities, dim=1)

    predictive_entropy = _safe_mean_entropy(mean_probabilities)
    expected_entropy = _safe_mean_entropy(probability_stack).mean(dim=0)
    mutual_information = predictive_entropy - expected_entropy

    if mean_probabilities.shape[1] == 2:
        probability_variance = probability_stack[:, :, 1].var(dim=0, unbiased=False)
    else:
        probability_variance = probability_stack.var(dim=0, unbiased=False).mean(dim=1)

    return {
        "mean_logits": mean_logits,
        "mean_probabilities": mean_probabilities,
        "predictions": predictions,
        "predictive_entropy": predictive_entropy,
        "probability_variance": probability_variance,
        "mutual_information": mutual_information,
    }


def _train_one_epoch(
    model: BayesianLinearClassifier,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    kl_weight: float,
    mc_samples_train: int,
    device: torch.device,
    show_progress: bool,
    progress_description: str,
) -> dict[str, float | int]:
    """Train the Bayesian head for one epoch."""

    total_loss = 0.0
    total_cross_entropy = 0.0
    total_kl = 0.0
    num_batches = 0
    num_examples = 0

    model.train()
    dataloader_iterator = tqdm(
        dataloader,
        desc=progress_description,
        disable=not show_progress,
        leave=False,
    )

    for batch in dataloader_iterator:
        features = batch["image"]
        labels = batch["label"]

        if not isinstance(features, torch.Tensor):
            raise TypeError("batch['image'] must be a torch.Tensor.")

        if not isinstance(labels, torch.Tensor):
            raise TypeError("batch['label'] must be a torch.Tensor.")

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        cross_entropy_losses = [
            F.cross_entropy(model(features, sample=True), labels) for _ in range(mc_samples_train)
        ]
        cross_entropy = torch.stack(cross_entropy_losses).mean()
        kl_value = model.kl_divergence()
        total_objective = cross_entropy + kl_weight * kl_value
        total_objective.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(total_objective.item()) * batch_size
        total_cross_entropy += float(cross_entropy.item()) * batch_size
        total_kl += float(kl_value.item()) * batch_size
        num_batches += 1
        num_examples += batch_size

        if show_progress:
            dataloader_iterator.set_postfix_str(
                f"loss={float(total_objective.item()):.4f}, ce={float(cross_entropy.item()):.4f}"
            )

    if num_batches == 0:
        raise ValueError("Cannot train for one epoch with an empty dataloader.")

    return {
        "loss": total_loss / num_examples,
        "cross_entropy": total_cross_entropy / num_examples,
        "kl": total_kl / num_examples,
        "num_batches": num_batches,
        "num_examples": num_examples,
    }


def _evaluate_bayesian_model(
    model: BayesianLinearClassifier,
    dataloader: Any,
    mc_samples_eval: int,
    device: torch.device,
    show_progress: bool,
    progress_description: str,
) -> dict[str, Any]:
    """Evaluate the Bayesian head with Monte Carlo predictive averaging."""

    total_loss = 0.0
    num_batches = 0
    num_examples = 0
    label_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []
    probability_batches: list[torch.Tensor] = []
    logit_batches: list[torch.Tensor] = []
    predictive_entropy_batches: list[torch.Tensor] = []
    probability_variance_batches: list[torch.Tensor] = []
    mutual_information_batches: list[torch.Tensor] = []
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

            predictive_stats = _predictive_statistics(
                model=model,
                features=features,
                mc_samples=mc_samples_eval,
            )
            mean_probabilities = predictive_stats["mean_probabilities"]
            mean_logits = predictive_stats["mean_logits"]
            predictions = predictive_stats["predictions"]

            selected_probabilities = mean_probabilities[
                torch.arange(labels.shape[0], device=labels.device),
                labels,
            ]
            batch_loss = float((-torch.log(torch.clamp(selected_probabilities, min=1e-12))).mean().item())
            batch_size = int(labels.shape[0])

            total_loss += batch_loss * batch_size
            num_batches += 1
            num_examples += batch_size

            label_batches.append(labels.detach().cpu())
            prediction_batches.append(predictions.detach().cpu())
            probability_batches.append(mean_probabilities.detach().cpu())
            logit_batches.append(mean_logits.detach().cpu())
            predictive_entropy_batches.append(
                predictive_stats["predictive_entropy"].detach().cpu()
            )
            probability_variance_batches.append(
                predictive_stats["probability_variance"].detach().cpu()
            )
            mutual_information_batches.append(
                predictive_stats["mutual_information"].detach().cpu()
            )
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
    probability_variance = torch.cat(probability_variance_batches, dim=0)
    mutual_information = torch.cat(mutual_information_batches, dim=0)

    metrics = classification_summary(
        predictions=predictions,
        labels=labels,
        num_classes=int(probabilities.shape[1]),
        positive_scores=probabilities[:, 1] if probabilities.shape[1] == 2 else None,
        probabilities=probabilities,
        logits=None,
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
        "probability_variance": probability_variance,
        "mutual_information": mutual_information,
        "image_paths": image_paths,
        "id_codes": id_codes,
        "metrics": metrics,
    }


def _write_validation_predictions_csv(
    validation_result: dict[str, Any],
    output_path: Path,
) -> None:
    """Write Bayesian predictive statistics to CSV."""

    probabilities = validation_result.get("probabilities")
    labels = validation_result.get("labels")
    predictions = validation_result.get("predictions")
    predictive_entropy = validation_result.get("predictive_entropy")
    probability_variance = validation_result.get("probability_variance")
    mutual_information = validation_result.get("mutual_information")
    image_paths = validation_result.get("image_paths")
    id_codes = validation_result.get("id_codes")

    if not isinstance(probabilities, torch.Tensor):
        raise ValueError("Validation result does not contain probability tensors.")

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            "validation_predictions.csv currently supports binary classification outputs only."
        )

    required_tensors = (
        labels,
        predictions,
        predictive_entropy,
        probability_variance,
        mutual_information,
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in required_tensors):
        raise ValueError("Validation result is missing predictive uncertainty tensors.")

    if not isinstance(image_paths, list) or not isinstance(id_codes, list):
        raise ValueError("Validation result is missing image path or id_code values.")

    confidence = torch.max(probabilities, dim=1).values
    dataframe = pd.DataFrame(
        {
            "id_code": id_codes,
            "image_path": image_paths,
            "true_label": labels.tolist(),
            "predicted_label": predictions.tolist(),
            "probability_class_0": probabilities[:, 0].tolist(),
            "probability_class_1": probabilities[:, 1].tolist(),
            "confidence": confidence.tolist(),
            "is_correct": (predictions == labels).tolist(),
            "predictive_entropy": predictive_entropy.tolist(),
            "probability_variance": probability_variance.tolist(),
            "mutual_information": mutual_information.tolist(),
        }
    )
    dataframe.to_csv(output_path, index=False)


def run_bayesian_feature_head_training(
    train_features: str | Path,
    val_features: str | Path,
    output_dir: str | Path,
    num_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str | None = "cpu",
    mc_samples_train: int = 1,
    mc_samples_eval: int = 10,
    prior_std: float = 1.0,
    kl_weight: float | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Train a Bayesian linear head on cached feature bundles."""

    if epochs <= 0:
        raise ValueError(f"epochs must be positive. Got: {epochs}")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive. Got: {batch_size}")

    if mc_samples_train <= 0:
        raise ValueError(f"mc_samples_train must be positive. Got: {mc_samples_train}")

    if mc_samples_eval <= 0:
        raise ValueError(f"mc_samples_eval must be positive. Got: {mc_samples_eval}")

    if prior_std <= 0.0:
        raise ValueError(f"prior_std must be positive. Got: {prior_std}")

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

    resolved_kl_weight = kl_weight
    if resolved_kl_weight is None:
        resolved_kl_weight = 1.0 / float(train_bundle["features"].shape[0])

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

    model = BayesianLinearClassifier(
        feature_dim=feature_dim,
        num_classes=num_classes,
        prior_std=prior_std,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_results: list[dict[str, Any]] = []
    final_validation_result: dict[str, Any] | None = None

    for epoch_index in range(epochs):
        epoch_number = epoch_index + 1
        train_result = _train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            kl_weight=resolved_kl_weight,
            mc_samples_train=mc_samples_train,
            device=resolved_device,
            show_progress=show_progress,
            progress_description=f"Train epoch {epoch_number}/{epochs}",
        )
        validation_result = _evaluate_bayesian_model(
            model=model,
            dataloader=val_dataloader,
            mc_samples_eval=mc_samples_eval,
            device=resolved_device,
            show_progress=show_progress,
            progress_description=f"Val epoch {epoch_number}/{epochs}",
        )
        final_validation_result = validation_result

        validation_metrics = validation_result["metrics"]
        epoch_result = {
            "epoch": epoch_number,
            "train_loss": float(train_result["loss"]),
            "train_cross_entropy": float(train_result["cross_entropy"]),
            "train_kl": float(train_result["kl"]),
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
            "prior_std": float(prior_std),
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
        "mc_samples_train": int(mc_samples_train),
        "mc_samples_eval": int(mc_samples_eval),
        "prior_std": float(prior_std),
        "kl_weight": float(resolved_kl_weight),
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
    """Parse cached-feature Bayesian-head training CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Train a Bayesian linear head on cached RETFound features."
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
    parser.add_argument("--mc-samples-train", type=int, default=1)
    parser.add_argument("--mc-samples-eval", type=int, default=10)
    parser.add_argument("--prior-std", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the cached-feature Bayesian-head training CLI."""

    args = parse_args()
    result = run_bayesian_feature_head_training(
        train_features=args.train_features,
        val_features=args.val_features,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        mc_samples_train=args.mc_samples_train,
        mc_samples_eval=args.mc_samples_eval,
        prior_std=args.prior_std,
        kl_weight=args.kl_weight,
        show_progress=not args.no_progress,
    )
    print(f"Saved metrics to: {result['metrics_path']}")


if __name__ == "__main__":
    main()
