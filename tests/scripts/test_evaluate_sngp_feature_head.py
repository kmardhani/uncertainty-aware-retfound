"""Tests for evaluation-only SNGP cached-feature head inference."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from scripts.evaluation.evaluate_sngp_feature_head import (
    main,
    run_sngp_feature_head_evaluation,
)
from uncertainty_retfound.models.sngp import SNGPFeatureClassifier


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


def _write_sngp_checkpoint_and_config(
    model_path: Path,
    config_path: Path,
) -> None:
    model = SNGPFeatureClassifier(
        input_dim=4,
        num_classes=2,
        hidden_dim=0,
        rff_dim=2,
        kernel_scale=1.0,
        ridge_penalty=1.0,
    )

    with torch.no_grad():
        model.random_features.random_weight.zero_()
        model.random_features.random_bias.zero_()
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.precision_diag.copy_(torch.tensor([2.0, 8.0], dtype=torch.float32))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            **model.to_serializable_config(),
        },
        model_path,
    )
    config_path.write_text(
        json.dumps(
            {
                "feature_dim": 4,
                "num_classes": 2,
                "hidden_dim": 0,
                "rff_dim": 2,
                "kernel_scale": 1.0,
                "ridge_penalty": 1.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_run_sngp_feature_head_evaluation_writes_expected_outputs(tmp_path: Path) -> None:
    features_path = tmp_path / "ddr_val_features.pt"
    model_path = tmp_path / "model.pt"
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "evaluation"

    _write_feature_bundle(
        features_path,
        features=[
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, -2.0, -3.0, -4.0],
        ],
        labels=[0, 1],
        id_codes=["ddr_a", "ddr_b"],
        image_paths=["/tmp/ddr_a.jpg", "/tmp/ddr_b.jpg"],
        split="val",
    )
    _write_sngp_checkpoint_and_config(model_path, config_path)

    result = run_sngp_feature_head_evaluation(
        features=features_path,
        model_path=model_path,
        config_json=config_path,
        output_dir=output_dir,
        batch_size=2,
        device="cpu",
        show_progress=False,
    )

    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.csv"
    metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)

    assert result["metrics_path"] == str(metrics_path)
    assert metrics_path.exists()
    assert predictions_path.exists()

    assert metrics_json["features"] == str(features_path)
    assert metrics_json["model_path"] == str(model_path)
    assert metrics_json["config_json"] == str(config_path)
    assert metrics_json["evaluated_split"] == "val"
    assert metrics_json["num_examples"] == 2
    assert "metrics" in metrics_json
    assert "accuracy" in metrics_json["metrics"]
    assert metrics_json["precision_diag_min"] == 2.0
    assert metrics_json["precision_diag_max"] == 8.0
    assert metrics_json["precision_diag_mean"] == 5.0

    assert list(predictions.columns) == [
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
    ]
    assert len(predictions) == 2
    assert set(predictions["id_code"]) == {"ddr_a", "ddr_b"}
    assert predictions["probability_class_0"].between(0.0, 1.0).all()
    assert predictions["probability_class_1"].between(0.0, 1.0).all()
    assert predictions["confidence"].between(0.0, 1.0).all()
    assert predictions["predictive_entropy"].map(math.isfinite).all()
    assert predictions["sngp_variance"].map(math.isfinite).all()
    assert predictions["sngp_uncertainty"].map(math.isfinite).all()

    expected_variance = 1.0 / 2.0 + 1.0 / 8.0
    assert predictions["sngp_variance"].tolist() == pytest.approx(
        [expected_variance, expected_variance]
    )
    expected_entropy = math.log(2.0)
    assert predictions["predictive_entropy"].tolist() == pytest.approx(
        [expected_entropy, expected_entropy]
    )
    assert predictions["sngp_uncertainty"].tolist() == pytest.approx(
        [expected_entropy + expected_variance, expected_entropy + expected_variance]
    )


def test_main_runs_evaluation_without_retraining(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    features_path = tmp_path / "ddr_test_features.pt"
    model_path = tmp_path / "model.pt"
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "evaluation"

    _write_feature_bundle(
        features_path,
        features=[[0.5, 0.5, 0.5, 0.5]],
        labels=[0],
        id_codes=["ddr_test_a"],
        image_paths=["/tmp/ddr_test_a.jpg"],
        split="test",
    )
    _write_sngp_checkpoint_and_config(model_path, config_path)

    main(
        [
            "--features",
            str(features_path),
            "--model-path",
            str(model_path),
            "--config-json",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "1",
            "--no-progress",
        ]
    )

    captured = capsys.readouterr()
    assert "Saved metrics to:" in captured.out
    assert "Saved predictions to:" in captured.out
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "predictions.csv").exists()
