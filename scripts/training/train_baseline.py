"""Run a small baseline image-classification training experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from uncertainty_retfound.data.aptos import APTOSDataset
from uncertainty_retfound.data.dataloaders import create_dataloader
from uncertainty_retfound.data.transforms import build_torchvision_transform_from_config
from uncertainty_retfound.evaluation.loops import evaluate_model
from uncertainty_retfound.models.baseline import SmallCNNClassifier
from uncertainty_retfound.models.retfound import build_retfound_linear_classifier
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


def _require_split_column(metadata: pd.DataFrame) -> None:
    """Validate that prepared metadata includes a split column."""

    if "split" not in metadata.columns:
        raise ValueError("Prepared metadata CSV must contain a 'split' column.")


def _build_model(
    model_type: str,
    num_classes: int,
    backbone_checkpoint: str | Path | None,
    feature_dim: int,
    freeze_encoder: bool,
) -> torch.nn.Module:
    """Build the requested classifier model."""

    if model_type == "small_cnn":
        return SmallCNNClassifier(num_classes=num_classes)

    if model_type == "retfound_linear":
        if backbone_checkpoint is None:
            raise ValueError(
                "backbone_checkpoint is required when model_type='retfound_linear'."
            )

        return build_retfound_linear_classifier(
            checkpoint_path=backbone_checkpoint,
            num_classes=num_classes,
            feature_dim=feature_dim,
            freeze_encoder=freeze_encoder,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def run_baseline_training(
    metadata_csv: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    num_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    resize: int | None,
    center_crop: int | None,
    num_workers: int,
    seed: int,
    device: str | None = "cpu",
    train_split: str = "train",
    val_split: str = "val",
    id_column: str = "id_code",
    label_column: str = "label",
    image_extension: str = ".png",
    show_progress: bool = True,
    save_batch_history: bool = True,
    model_type: str = "small_cnn",
    backbone_checkpoint: str | Path | None = None,
    feature_dim: int = 1024,
    freeze_encoder: bool = True,
) -> dict[str, Any]:
    """Run a baseline training experiment and write JSON metrics."""

    if epochs <= 0:
        raise ValueError(f"epochs must be positive. Got: {epochs}")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive. Got: {batch_size}")

    _set_seed(seed)

    metadata_path = Path(metadata_csv)
    image_root_path = Path(image_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    _require_split_column(metadata)

    train_metadata = metadata.loc[metadata["split"] == train_split].copy()
    val_metadata = metadata.loc[metadata["split"] == val_split].copy()

    if train_metadata.empty:
        raise ValueError(f"No rows found for train split '{train_split}'.")

    if val_metadata.empty:
        raise ValueError(f"No rows found for validation split '{val_split}'.")

    transform = build_torchvision_transform_from_config(
        {
            "resize": resize,
            "center_crop": center_crop,
            "to_tensor": True,
        }
    )

    train_dataset = APTOSDataset(
        metadata=train_metadata,
        image_root=image_root_path,
        id_column=id_column,
        label_column=label_column,
        image_extension=image_extension,
        transform=transform,
    )
    val_dataset = APTOSDataset(
        metadata=val_metadata,
        image_root=image_root_path,
        id_column=id_column,
        label_column=label_column,
        image_extension=image_extension,
        transform=transform,
    )

    resolved_device = _resolve_device(device)
    pin_memory = resolved_device.type != "cpu"

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = _build_model(
        model_type=model_type,
        num_classes=num_classes,
        backbone_checkpoint=backbone_checkpoint,
        feature_dim=feature_dim,
        freeze_encoder=freeze_encoder,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_results: list[dict[str, Any]] = []
    batch_history: dict[str, list[dict[str, float | int]]] = {
        "train": [],
        "validation": [],
    }

    for epoch_index in range(epochs):
        epoch_number = epoch_index + 1
        train_result = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=resolved_device,
            show_progress=show_progress,
            progress_description=f"Train epoch {epoch_number}/{epochs}",
            include_batch_history=save_batch_history,
        )
        validation_result = evaluate_model(
            model=model,
            dataloader=val_dataloader,
            device=resolved_device,
            include_metrics=True,
            show_progress=show_progress,
            progress_description=f"Val epoch {epoch_number}/{epochs}",
            include_batch_history=save_batch_history,
        )

        validation_metrics = validation_result["metrics"]
        epoch_result = {
            "epoch": epoch_number,
            "train_loss": float(train_result["loss"]),
            "val_loss": float(validation_result["loss"]),
            "val_accuracy": float(validation_metrics["accuracy"]),
            "val_metrics": validation_metrics,
        }
        epoch_results.append(epoch_result)

        if save_batch_history:
            train_batch_history = train_result.get("batch_history", [])
            validation_batch_history = validation_result.get("batch_history", [])

            batch_history["train"].extend(
                {"epoch": epoch_number, **batch_record} for batch_record in train_batch_history
            )
            batch_history["validation"].extend(
                {"epoch": epoch_number, **batch_record}
                for batch_record in validation_batch_history
            )

        print(
            f"Epoch {epoch_number}/{epochs} "
            f"train_loss={epoch_result['train_loss']:.4f} "
            f"val_loss={epoch_result['val_loss']:.4f} "
            f"val_accuracy={epoch_result['val_accuracy']:.4f}"
        )

    final_result = {
        "metadata_csv": str(metadata_path),
        "image_root": str(image_root_path),
        "output_dir": str(output_dir_path),
        "num_classes": int(num_classes),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "train_split": train_split,
        "validation_split": val_split,
        "device": str(resolved_device),
        "model_type": model_type,
        "epoch_results": epoch_results,
        "final_validation_metrics": epoch_results[-1]["val_metrics"],
    }

    if backbone_checkpoint is not None:
        final_result["backbone_checkpoint"] = str(backbone_checkpoint)

    if model_type == "retfound_linear":
        final_result["feature_dim"] = int(feature_dim)
        final_result["freeze_encoder"] = bool(freeze_encoder)

    if save_batch_history:
        final_result["batch_history"] = batch_history

    metrics_path = output_dir_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(_serialize_for_json(final_result), indent=2),
        encoding="utf-8",
    )
    final_result["metrics_path"] = str(metrics_path)

    return final_result


def parse_args() -> argparse.Namespace:
    """Parse baseline training CLI arguments."""

    parser = argparse.ArgumentParser(description="Train a small baseline image classifier.")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--center-crop", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--id-column", type=str, default="id_code")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--image-extension", type=str, default=".png")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("small_cnn", "retfound_linear"),
        default="small_cnn",
    )
    parser.add_argument("--backbone-checkpoint", type=Path, default=None)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true")
    parser.add_argument("--unfreeze-encoder", dest="freeze_encoder", action="store_false")
    parser.set_defaults(freeze_encoder=True)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-save-batch-history", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the baseline training CLI."""

    args = parse_args()
    result = run_baseline_training(
        metadata_csv=args.metadata_csv,
        image_root=args.image_root,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resize=args.resize,
        center_crop=args.center_crop,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        train_split=args.train_split,
        val_split=args.val_split,
        id_column=args.id_column,
        label_column=args.label_column,
        image_extension=args.image_extension,
        show_progress=not args.no_progress,
        save_batch_history=not args.no_save_batch_history,
        model_type=args.model_type,
        backbone_checkpoint=args.backbone_checkpoint,
        feature_dim=args.feature_dim,
        freeze_encoder=args.freeze_encoder,
    )
    print(f"Saved metrics to: {result['metrics_path']}")


if __name__ == "__main__":
    main()
