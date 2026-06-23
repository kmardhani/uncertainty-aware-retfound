"""Tests for the baseline training CLI logic."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from scripts.training.train_baseline import run_baseline_training


def _write_fake_png(image_path: Path, color: tuple[int, int, int]) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), color=color)
    image.save(image_path)


def _make_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_code": ["sample_a", "sample_b", "sample_c", "sample_d"],
            "label": [0, 1, 0, 1],
            "split": ["train", "train", "val", "val"],
        }
    )


def test_run_baseline_training_writes_metrics_json(tmp_path: Path) -> None:
    metadata = _make_metadata()
    metadata_path = tmp_path / "prepared_metadata.csv"
    image_root = tmp_path / "train_images"
    output_dir = tmp_path / "outputs"

    _write_fake_png(image_root / "sample_a.png", color=(255, 0, 0))
    _write_fake_png(image_root / "sample_b.png", color=(0, 255, 0))
    _write_fake_png(image_root / "sample_c.png", color=(0, 0, 255))
    _write_fake_png(image_root / "sample_d.png", color=(255, 255, 0))
    metadata.to_csv(metadata_path, index=False)

    result = run_baseline_training(
        metadata_csv=metadata_path,
        image_root=image_root,
        output_dir=output_dir,
        num_classes=2,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        resize=32,
        center_crop=32,
        num_workers=0,
        seed=42,
        show_progress=False,
    )

    metrics_path = output_dir / "metrics.json"
    metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metrics_path.exists()
    assert result["metrics_path"] == str(metrics_path)
    assert metrics_json["metadata_csv"] == str(metadata_path)
    assert metrics_json["image_root"] == str(image_root)
    assert metrics_json["num_classes"] == 2
    assert metrics_json["epochs"] == 1
    assert metrics_json["batch_size"] == 2
    assert metrics_json["learning_rate"] == 1e-3
    assert metrics_json["train_split"] == "train"
    assert metrics_json["validation_split"] == "val"
    assert len(metrics_json["epoch_results"]) == 1
    assert "batch_history" in metrics_json
    assert len(metrics_json["batch_history"]["train"]) >= 1
    assert len(metrics_json["batch_history"]["validation"]) >= 1

    epoch_result = metrics_json["epoch_results"][0]
    assert epoch_result["epoch"] == 1
    assert epoch_result["train_loss"] >= 0.0
    assert epoch_result["val_loss"] >= 0.0
    assert 0.0 <= epoch_result["val_accuracy"] <= 1.0
    assert "accuracy" in epoch_result["val_metrics"]
    assert "confusion_matrix" in metrics_json["final_validation_metrics"]

    train_batch_record = metrics_json["batch_history"]["train"][0]
    validation_batch_record = metrics_json["batch_history"]["validation"][0]

    assert set(train_batch_record) == {"epoch", "batch", "loss", "num_examples"}
    assert train_batch_record["epoch"] == 1
    assert train_batch_record["batch"] >= 1
    assert math.isfinite(train_batch_record["loss"])
    assert train_batch_record["loss"] >= 0.0
    assert train_batch_record["num_examples"] > 0

    assert set(validation_batch_record) == {
        "epoch",
        "batch",
        "loss",
        "accuracy",
        "num_examples",
    }
    assert validation_batch_record["epoch"] == 1
    assert validation_batch_record["batch"] >= 1
    assert math.isfinite(validation_batch_record["loss"])
    assert validation_batch_record["loss"] >= 0.0
    assert math.isfinite(validation_batch_record["accuracy"])
    assert 0.0 <= validation_batch_record["accuracy"] <= 1.0
    assert validation_batch_record["num_examples"] > 0


def test_run_baseline_training_requires_split_column(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "id_code": ["sample_a", "sample_b"],
            "label": [0, 1],
        }
    )
    metadata_path = tmp_path / "prepared_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="must contain a 'split' column"):
        run_baseline_training(
            metadata_csv=metadata_path,
            image_root=tmp_path / "train_images",
            output_dir=tmp_path / "outputs",
            num_classes=2,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            resize=32,
            center_crop=32,
            num_workers=0,
            seed=42,
            show_progress=False,
        )


@pytest.mark.parametrize(
    ("splits", "expected_error"),
    [
        (["train", "train"], "validation split 'val'"),
        (["val", "val"], "train split 'train'"),
    ],
)
def test_run_baseline_training_rejects_empty_train_or_validation_split(
    tmp_path: Path,
    splits: list[str],
    expected_error: str,
) -> None:
    metadata = pd.DataFrame(
        {
            "id_code": ["sample_a", "sample_b"],
            "label": [0, 1],
            "split": splits,
        }
    )
    metadata_path = tmp_path / "prepared_metadata.csv"
    image_root = tmp_path / "train_images"

    _write_fake_png(image_root / "sample_a.png", color=(255, 0, 0))
    _write_fake_png(image_root / "sample_b.png", color=(0, 255, 0))
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match=expected_error):
        run_baseline_training(
            metadata_csv=metadata_path,
            image_root=image_root,
            output_dir=tmp_path / "outputs",
            num_classes=2,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            resize=32,
            center_crop=32,
            num_workers=0,
            seed=42,
            show_progress=False,
        )


def test_run_baseline_training_can_disable_batch_history(tmp_path: Path) -> None:
    metadata = _make_metadata()
    metadata_path = tmp_path / "prepared_metadata.csv"
    image_root = tmp_path / "train_images"
    output_dir = tmp_path / "outputs"

    _write_fake_png(image_root / "sample_a.png", color=(255, 0, 0))
    _write_fake_png(image_root / "sample_b.png", color=(0, 255, 0))
    _write_fake_png(image_root / "sample_c.png", color=(0, 0, 255))
    _write_fake_png(image_root / "sample_d.png", color=(255, 255, 0))
    metadata.to_csv(metadata_path, index=False)

    result = run_baseline_training(
        metadata_csv=metadata_path,
        image_root=image_root,
        output_dir=output_dir,
        num_classes=2,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        resize=32,
        center_crop=32,
        num_workers=0,
        seed=42,
        show_progress=False,
        save_batch_history=False,
    )

    metrics_json = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert "batch_history" not in metrics_json
    assert "batch_history" not in result
