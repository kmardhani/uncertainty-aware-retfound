"""Tests for training an SNGP-style cached-feature head."""

from __future__ import annotations

import math
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from scripts.training.train_sngp_feature_head import run_sngp_feature_head_training


def _write_feature_bundle(
    output_path: Path,
    *,
    features: list[list[float]],
    labels: list[int],
    id_codes: list[str],
    image_paths: list[str],
    split: str,
) -> None:
    torch.save(
        {
            "features": torch.tensor(features, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "id_codes": id_codes,
            "image_paths": image_paths,
            "split": split,
        },
        output_path,
    )


def test_run_sngp_feature_head_training_writes_expected_outputs(tmp_path: Path) -> None:
    train_features_path = tmp_path / "train_features.pt"
    val_features_path = tmp_path / "val_features.pt"
    output_dir = tmp_path / "outputs"

    _write_feature_bundle(
        train_features_path,
        features=[
            [2.0, 2.0, 2.0, 2.0],
            [1.5, 1.0, 1.0, 1.2],
            [-2.0, -2.0, -2.0, -2.0],
            [-1.5, -1.0, -1.0, -1.2],
        ],
        labels=[1, 1, 0, 0],
        id_codes=["train_a", "train_b", "train_c", "train_d"],
        image_paths=[
            "/tmp/train_a.png",
            "/tmp/train_b.png",
            "/tmp/train_c.png",
            "/tmp/train_d.png",
        ],
        split="train",
    )
    _write_feature_bundle(
        val_features_path,
        features=[
            [1.2, 1.1, 1.0, 1.3],
            [-1.2, -1.1, -1.0, -1.3],
        ],
        labels=[1, 0],
        id_codes=["val_a", "val_b"],
        image_paths=["/tmp/val_a.png", "/tmp/val_b.png"],
        split="val",
    )

    result = run_sngp_feature_head_training(
        train_features=train_features_path,
        val_features=val_features_path,
        output_dir=output_dir,
        num_classes=2,
        epochs=3,
        batch_size=2,
        learning_rate=1e-2,
        seed=42,
        device="cpu",
        hidden_dim=8,
        rff_dim=16,
        kernel_scale=1.0,
        ridge_penalty=0.5,
        selection_metric="val_loss",
        show_progress=False,
    )

    metrics_path = output_dir / "metrics.json"
    best_metrics_path = output_dir / "best_metrics.json"
    config_path = output_dir / "config.json"
    model_path = output_dir / "model.pt"
    best_model_path = output_dir / "best_model.pt"
    validation_predictions_path = output_dir / "validation_predictions.csv"
    best_validation_predictions_path = output_dir / "best_validation_predictions.csv"

    metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))
    best_metrics_json = json.loads(best_metrics_path.read_text(encoding="utf-8"))
    config_json = json.loads(config_path.read_text(encoding="utf-8"))
    validation_predictions = pd.read_csv(validation_predictions_path)
    best_validation_predictions = pd.read_csv(best_validation_predictions_path)
    model_bundle = torch.load(model_path, map_location="cpu")
    best_model_bundle = torch.load(best_model_path, map_location="cpu")

    assert result["metrics_path"] == str(metrics_path)
    assert result["config_path"] == str(config_path)
    assert metrics_path.exists()
    assert best_metrics_path.exists()
    assert config_path.exists()
    assert model_path.exists()
    assert best_model_path.exists()
    assert validation_predictions_path.exists()
    assert best_validation_predictions_path.exists()

    assert metrics_json["num_classes"] == 2
    assert metrics_json["epochs"] == 3
    assert metrics_json["batch_size"] == 2
    assert metrics_json["hidden_dim"] == 8
    assert metrics_json["rff_dim"] == 16
    assert metrics_json["kernel_scale"] == 1.0
    assert metrics_json["ridge_penalty"] == 0.5
    assert metrics_json["best_epoch"] >= 1
    assert metrics_json["best_metric_name"] == "val_loss"
    assert isinstance(metrics_json["best_metric_value"], float)
    assert len(metrics_json["epoch_results"]) == 3
    assert "final_validation_metrics" in metrics_json
    assert "accuracy" in metrics_json["final_validation_metrics"]
    assert "auc" in metrics_json["final_validation_metrics"]
    assert "expected_calibration_error" in metrics_json["final_validation_metrics"]
    assert best_metrics_json["best_metric_name"] == "val_loss"
    assert best_metrics_json["best_epoch"] == metrics_json["best_epoch"]

    assert config_json["feature_dim"] == 4
    assert config_json["train_features"] == str(train_features_path)
    assert config_json["val_features"] == str(val_features_path)
    assert config_json["best_metric_name"] == "val_loss"
    assert config_json["best_epoch"] == metrics_json["best_epoch"]

    assert model_bundle["input_dim"] == 4
    assert model_bundle["num_classes"] == 2
    assert model_bundle["hidden_dim"] == 8
    assert model_bundle["rff_dim"] == 16
    assert "model_state_dict" in model_bundle
    assert best_model_bundle["input_dim"] == 4
    assert best_model_bundle["num_classes"] == 2
    assert "model_state_dict" in best_model_bundle

    required_columns = {
        "id_code",
        "image_path",
        "true_label",
        "predicted_label",
        "logit_class_0",
        "logit_class_1",
        "probability_class_0",
        "probability_class_1",
        "confidence",
        "is_correct",
        "predictive_entropy",
        "sngp_variance",
        "sngp_uncertainty",
    }
    assert required_columns.issubset(validation_predictions.columns)
    assert len(validation_predictions) == 2
    assert set(validation_predictions["id_code"]) == {"val_a", "val_b"}
    assert validation_predictions["probability_class_0"].between(0.0, 1.0).all()
    assert validation_predictions["probability_class_1"].between(0.0, 1.0).all()
    assert validation_predictions["confidence"].between(0.0, 1.0).all()
    assert validation_predictions["predictive_entropy"].map(math.isfinite).all()
    assert validation_predictions["sngp_variance"].map(math.isfinite).all()
    assert validation_predictions["sngp_uncertainty"].map(math.isfinite).all()
    assert list(best_validation_predictions.columns) == list(validation_predictions.columns)
    assert len(best_validation_predictions) == len(validation_predictions)


def test_run_sngp_feature_head_training_supports_selection_metric(tmp_path: Path) -> None:
    train_features_path = tmp_path / "train_features.pt"
    val_features_path = tmp_path / "val_features.pt"
    output_dir = tmp_path / "outputs"

    _write_feature_bundle(
        train_features_path,
        features=[
            [2.0, 2.0, 2.0, 2.0],
            [1.5, 1.0, 1.0, 1.2],
            [-2.0, -2.0, -2.0, -2.0],
            [-1.5, -1.0, -1.0, -1.2],
        ],
        labels=[1, 1, 0, 0],
        id_codes=["train_a", "train_b", "train_c", "train_d"],
        image_paths=[
            "/tmp/train_a.png",
            "/tmp/train_b.png",
            "/tmp/train_c.png",
            "/tmp/train_d.png",
        ],
        split="train",
    )
    _write_feature_bundle(
        val_features_path,
        features=[
            [1.2, 1.1, 1.0, 1.3],
            [-1.2, -1.1, -1.0, -1.3],
        ],
        labels=[1, 0],
        id_codes=["val_a", "val_b"],
        image_paths=["/tmp/val_a.png", "/tmp/val_b.png"],
        split="val",
    )

    result = run_sngp_feature_head_training(
        train_features=train_features_path,
        val_features=val_features_path,
        output_dir=output_dir,
        num_classes=2,
        epochs=2,
        batch_size=2,
        learning_rate=1e-2,
        seed=42,
        device="cpu",
        hidden_dim=4,
        rff_dim=8,
        kernel_scale=1.0,
        ridge_penalty=1.0,
        selection_metric="balanced_accuracy",
        show_progress=False,
    )

    metrics_json = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    config_json = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    best_metrics_json = json.loads((output_dir / "best_metrics.json").read_text(encoding="utf-8"))

    assert result["best_metrics"]["best_metric_name"] == "balanced_accuracy"
    assert metrics_json["best_metric_name"] == "balanced_accuracy"
    assert config_json["best_metric_name"] == "balanced_accuracy"
    assert best_metrics_json["best_metric_name"] == "balanced_accuracy"
    assert isinstance(metrics_json["best_metric_value"], float)
    assert 1 <= metrics_json["best_epoch"] <= 2


def test_run_sngp_feature_head_training_rejects_invalid_ridge_penalty(
    tmp_path: Path,
) -> None:
    train_features_path = tmp_path / "train_features.pt"
    val_features_path = tmp_path / "val_features.pt"

    _write_feature_bundle(
        train_features_path,
        features=[[1.0, 2.0, 3.0, 4.0]],
        labels=[1],
        id_codes=["train_a"],
        image_paths=["/tmp/train_a.png"],
        split="train",
    )
    _write_feature_bundle(
        val_features_path,
        features=[[1.0, 2.0, 3.0, 4.0]],
        labels=[1],
        id_codes=["val_a"],
        image_paths=["/tmp/val_a.png"],
        split="val",
    )

    with pytest.raises(ValueError, match="ridge_penalty must be positive"):
        run_sngp_feature_head_training(
            train_features=train_features_path,
            val_features=val_features_path,
            output_dir=tmp_path / "outputs",
            num_classes=2,
            epochs=1,
            batch_size=1,
            learning_rate=1e-2,
            seed=42,
            device="cpu",
            hidden_dim=0,
            rff_dim=8,
            kernel_scale=1.0,
            ridge_penalty=0.0,
            show_progress=False,
        )
