"""Export frozen RETFound encoder features for prepared metadata splits."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from uncertainty_retfound.data.dataloaders import create_dataloader
from uncertainty_retfound.data.metadata_dataset import MetadataImageDataset
from uncertainty_retfound.data.transforms import build_torchvision_transform_from_config
from uncertainty_retfound.models.retfound import (
    _extract_features_from_encoder_output,
    load_external_retfound_encoder,
)


def _resolve_device(device: str | None) -> torch.device:
    """Resolve a requested device into a concrete torch device."""

    if device in (None, "cpu"):
        return torch.device("cpu")

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def _require_split_column(metadata: pd.DataFrame) -> None:
    """Validate that prepared metadata includes a split column."""

    if "split" not in metadata.columns:
        raise ValueError("Prepared metadata CSV must contain a 'split' column.")


def _extract_batch_string_values(batch: Mapping[str, object], key: str) -> list[str]:
    """Extract string-like values from a batch mapping."""

    raw_values = batch.get(key)
    if raw_values is None:
        return []

    if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
        raise TypeError(f"batch['{key}'] must be a list or tuple of strings.")

    sequence_values = cast(Sequence[object], raw_values)
    return [str(raw_value) for raw_value in sequence_values]


def _extract_features_for_dataloader(
    encoder: nn.Module,
    dataloader: DataLoader[Any],
    feature_dim: int,
    device: torch.device,
    show_progress: bool,
    progress_description: str,
) -> dict[str, Any]:
    """Run a frozen encoder over a dataloader and collect split features."""

    encoder = encoder.to(device)
    encoder.eval()

    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    id_codes: list[str] = []
    image_paths: list[str] = []

    dataloader_iterator = tqdm(
        dataloader,
        desc=progress_description,
        disable=not show_progress,
        leave=False,
    )

    with torch.no_grad():
        for batch in dataloader_iterator:
            if not isinstance(batch, Mapping):
                raise TypeError("Each dataloader batch must be a mapping.")

            batch_mapping = cast(Mapping[str, object], batch)
            images_object: object = batch_mapping["image"]
            labels_object: object = batch_mapping["label"]

            if not isinstance(images_object, torch.Tensor):
                raise TypeError("batch['image'] must be a torch.Tensor.")

            if not isinstance(labels_object, torch.Tensor):
                raise TypeError("batch['label'] must be a torch.Tensor.")

            images = images_object.to(device)
            labels = labels_object.to(dtype=torch.int64)

            encoder_output = encoder(images)
            features = _extract_features_from_encoder_output(
                encoder_output=encoder_output,
                feature_dim=feature_dim,
            )

            feature_batches.append(features.detach().cpu())
            label_batches.append(labels.detach().cpu())
            id_codes.extend(_extract_batch_string_values(batch_mapping, "id_code"))
            image_paths.extend(_extract_batch_string_values(batch_mapping, "image_path"))

    if not feature_batches:
        raise ValueError("Cannot export features for an empty dataloader.")

    return {
        "features": torch.cat(feature_batches, dim=0),
        "labels": torch.cat(label_batches, dim=0),
        "id_codes": id_codes,
        "image_paths": image_paths,
    }


def _serialize_for_json(value: Any) -> Any:
    """Convert nested outputs into JSON-serializable values."""

    if isinstance(value, torch.Tensor):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {key: _serialize_for_json(nested_value) for key, nested_value in value.items()}

    if isinstance(value, list):
        return [_serialize_for_json(item) for item in value]

    return value


def _export_split_features(
    encoder: nn.Module,
    metadata: pd.DataFrame,
    image_root: Path,
    output_path: Path,
    split_name: str,
    feature_dim: int,
    batch_size: int,
    num_workers: int,
    id_column: str,
    label_column: str,
    transform: Any,
    device: torch.device,
    show_progress: bool,
) -> int:
    """Export one split to a `.pt` feature bundle and return the example count."""

    split_metadata = metadata.loc[metadata["split"] == split_name].copy()
    if split_metadata.empty:
        raise ValueError(f"No rows found for split '{split_name}'.")

    dataset = MetadataImageDataset(
        metadata=split_metadata,
        image_root=image_root,
        id_column=id_column,
        label_column=label_column,
        transform=transform,
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type != "cpu",
    )
    split_result = _extract_features_for_dataloader(
        encoder=encoder,
        dataloader=dataloader,
        feature_dim=feature_dim,
        device=device,
        show_progress=show_progress,
        progress_description=f"Export {split_name} features",
    )
    split_result["split"] = split_name
    torch.save(split_result, output_path)
    return int(split_result["features"].shape[0])


def run_feature_export(
    metadata_csv: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    backbone_checkpoint: str | Path,
    retfound_repo_path: str | Path,
    batch_size: int = 32,
    resize: int | None = None,
    center_crop: int | None = None,
    num_workers: int = 0,
    device: str | None = "cpu",
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
    id_column: str = "id_code",
    label_column: str = "label",
    image_extension: str = ".png",
    feature_dim: int = 1024,
    architecture: str = "RETFound_mae",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Export frozen RETFound features for train/val/test splits."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive. Got: {batch_size}")

    metadata_path = Path(metadata_csv)
    image_root_path = Path(image_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    _require_split_column(metadata)

    transform = build_torchvision_transform_from_config(
        {
            "resize": resize,
            "center_crop": center_crop,
            "to_tensor": True,
        }
    )
    resolved_device = _resolve_device(device)
    encoder = load_external_retfound_encoder(
        retfound_repo_path=retfound_repo_path,
        checkpoint_path=backbone_checkpoint,
        architecture=architecture,
        feature_dim=feature_dim,
    )

    split_output_paths = {
        "train": output_dir_path / "train_features.pt",
        "val": output_dir_path / "val_features.pt",
        "test": output_dir_path / "test_features.pt",
    }
    split_names = {
        "train": train_split,
        "val": val_split,
        "test": test_split,
    }

    counts: dict[str, int] = {}
    for split_key, split_name in split_names.items():
        counts[split_key] = _export_split_features(
            encoder=encoder,
            metadata=metadata,
            image_root=image_root_path,
            output_path=split_output_paths[split_key],
            split_name=split_name,
            feature_dim=feature_dim,
            batch_size=batch_size,
            num_workers=num_workers,
            id_column=id_column,
            label_column=label_column,
            transform=transform,
            device=resolved_device,
            show_progress=show_progress,
        )

    feature_metadata = {
        "model": architecture,
        "backbone_checkpoint": str(backbone_checkpoint),
        "retfound_repo_path": str(retfound_repo_path),
        "metadata_csv": str(metadata_path),
        "image_root": str(image_root_path),
        "output_dir": str(output_dir_path),
        "feature_dim": int(feature_dim),
        "resize": resize,
        "center_crop": center_crop,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "device": str(resolved_device),
        "counts": counts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path_out = output_dir_path / "feature_metadata.json"
    metadata_path_out.write_text(
        json.dumps(_serialize_for_json(feature_metadata), indent=2),
        encoding="utf-8",
    )

    return {
        "feature_metadata_path": str(metadata_path_out),
        "train_features_path": str(split_output_paths["train"]),
        "val_features_path": str(split_output_paths["val"]),
        "test_features_path": str(split_output_paths["test"]),
        "counts": counts,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse feature export CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Export frozen RETFound features for prepared metadata."
    )
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--retfound-repo-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--center-crop", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--id-column", type=str, default="id_code")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--image-extension", type=str, default=".png")
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--architecture", type=str, default="RETFound_mae")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the RETFound feature export CLI."""

    args = parse_args(argv)
    result = run_feature_export(
        metadata_csv=args.metadata_csv,
        image_root=args.image_root,
        output_dir=args.output_dir,
        backbone_checkpoint=args.backbone_checkpoint,
        retfound_repo_path=args.retfound_repo_path,
        batch_size=args.batch_size,
        resize=args.resize,
        center_crop=args.center_crop,
        num_workers=args.num_workers,
        device=args.device,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        id_column=args.id_column,
        label_column=args.label_column,
        image_extension=args.image_extension,
        feature_dim=args.feature_dim,
        architecture=args.architecture,
        show_progress=not args.no_progress,
    )
    print(f"Saved feature metadata to: {result['feature_metadata_path']}")


if __name__ == "__main__":
    main()
