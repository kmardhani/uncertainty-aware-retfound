"""Tests for the baseline run summary CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.summarize_baseline_run import main, summarize_baseline_run


def _write_metrics_json(output_dir: Path) -> None:
    metrics = {
        "model_type": "small_cnn",
        "output_dir": str(output_dir),
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.001,
        "device": "cpu",
        "backbone_checkpoint": "/tmp/backbone.pth",
        "retfound_repo_path": "/tmp/retfound_repo",
        "epoch_results": [
            {
                "epoch": 1,
                "train_loss": 0.6765,
                "val_loss": 0.6794,
                "val_accuracy": 0.5792,
                "val_metrics": {
                    "accuracy": 0.5792,
                    "auc": 0.6123,
                    "precision": 0.6000,
                    "recall": 0.5500,
                    "sensitivity": 0.5500,
                    "specificity": 0.6100,
                    "f1": 0.5739,
                    "balanced_accuracy": 0.5800,
                    "brier_score": 0.2420,
                    "negative_log_likelihood": 0.6794,
                    "expected_calibration_error": 0.0810,
                    "mean_confidence": 0.6400,
                    "mean_positive_class_probability": 0.4700,
                },
            },
            {
                "epoch": 2,
                "train_loss": 0.6400,
                "val_loss": 0.6500,
                "val_accuracy": 0.7000,
                "val_metrics": {
                    "accuracy": 0.7000,
                    "auc": 0.7400,
                    "precision": 0.7200,
                    "recall": 0.6800,
                    "sensitivity": 0.6800,
                    "specificity": 0.7200,
                    "f1": 0.6994,
                    "balanced_accuracy": 0.7000,
                    "brier_score": 0.2100,
                    "negative_log_likelihood": 0.6500,
                    "expected_calibration_error": 0.0400,
                    "mean_confidence": 0.7100,
                    "mean_positive_class_probability": 0.5000,
                },
            },
        ],
        "final_validation_metrics": {
            "accuracy": 0.7000,
            "auc": 0.7400,
            "precision": 0.7200,
            "recall": 0.6800,
            "sensitivity": 0.6800,
            "specificity": 0.7200,
            "f1": 0.6994,
            "balanced_accuracy": 0.7000,
            "brier_score": 0.2100,
            "negative_log_likelihood": 0.6500,
            "expected_calibration_error": 0.0400,
            "mean_confidence": 0.7100,
            "mean_positive_class_probability": 0.5000,
            "confusion_matrix": [[7, 3], [3, 7]],
            "per_class_accuracy": {"0": 0.7, "1": 0.7},
            "num_classes": 2,
            "num_examples": 20,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _write_validation_predictions_csv(output_dir: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "id_code": ["sample_a", "sample_b", "sample_c"],
            "image_path": [
                "/tmp/images/sample_a.png",
                "/tmp/images/sample_b.png",
                "/tmp/images/sample_c.png",
            ],
            "true_label": [0, 1, 1],
            "predicted_label": [0, 1, 0],
            "probability_class_0": [0.8, 0.2, 0.7],
            "probability_class_1": [0.2, 0.8, 0.3],
            "confidence": [0.8, 0.8, 0.7],
            "is_correct": [True, True, False],
        }
    )
    dataframe.to_csv(output_dir / "validation_predictions.csv", index=False)


def test_summarize_baseline_run_includes_metrics_and_predictions(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_run"
    output_dir.mkdir(parents=True)
    _write_metrics_json(output_dir)
    _write_validation_predictions_csv(output_dir)

    summary = summarize_baseline_run(output_dir)

    assert "Run Configuration" in summary
    assert "- model_type: small_cnn" in summary
    assert "- output_dir:" in summary
    assert "- backbone_checkpoint: /tmp/backbone.pth" in summary
    assert "- retfound_repo_path: /tmp/retfound_repo" in summary
    assert "Epoch Results" in summary
    assert "recall/sensitivity" in summary
    assert "expected_calibration_error" in summary
    assert "Best Epochs" in summary
    assert "Highest accuracy: epoch 2 (val_accuracy=0.7000)" in summary
    assert "Lowest validation loss: epoch 2 (val_loss=0.6500)" in summary
    assert "Highest recall/sensitivity: epoch 2 (sensitivity=0.6800)" in summary
    assert "Highest balanced_accuracy: epoch 2 (balanced_accuracy=0.7000)" in summary
    assert "Lowest expected_calibration_error: epoch 2 (expected_calibration_error=0.0400)" in summary
    assert "Final Validation Metrics" in summary
    assert '"accuracy": 0.7' in summary
    assert "Validation Predictions" in summary
    assert "- row_count: 3" in summary
    assert "- columns: id_code, image_path, true_label, predicted_label" in summary
    assert "- correct: 2" in summary
    assert "- incorrect: 1" in summary
    assert "- average_confidence: 0.7667" in summary


def test_summarize_baseline_run_notes_missing_predictions_csv(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_run"
    output_dir.mkdir(parents=True)
    _write_metrics_json(output_dir)

    summary = summarize_baseline_run(output_dir)

    assert "Validation Predictions" in summary
    assert "validation_predictions.csv not found." in summary


def test_summarize_baseline_run_requires_metrics_json(tmp_path: Path) -> None:
    output_dir = tmp_path / "baseline_run"
    output_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="metrics.json not found"):
        summarize_baseline_run(output_dir)


def test_main_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "baseline_run"
    output_dir.mkdir(parents=True)
    _write_metrics_json(output_dir)
    _write_validation_predictions_csv(output_dir)

    main(["--output-dir", str(output_dir)])

    captured = capsys.readouterr()
    assert "Run Configuration" in captured.out
    assert "Epoch Results" in captured.out
    assert "Validation Predictions" in captured.out
