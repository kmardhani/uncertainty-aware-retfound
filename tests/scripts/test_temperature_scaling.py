"""Tests for the temperature scaling analysis utility."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.temperature_scaling import main, run_temperature_scaling


def _write_predictions_csv(predictions_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "id_code": ["sample_a", "sample_b", "sample_c", "sample_d"],
            "image_path": [
                "/tmp/images/sample_a.png",
                "/tmp/images/sample_b.png",
                "/tmp/images/sample_c.png",
                "/tmp/images/sample_d.png",
            ],
            "true_label": [0, 1, 0, 1],
            "predicted_label": [0, 1, 0, 1],
            "logit_class_0": [2.2, -0.6, 1.4, -0.2],
            "logit_class_1": [0.1, 1.8, -0.1, 0.7],
            "probability_class_0": [0.8909, 0.0832, 0.8176, 0.2891],
            "probability_class_1": [0.1091, 0.9168, 0.1824, 0.7109],
            "confidence": [0.8909, 0.9168, 0.8176, 0.7109],
            "is_correct": [True, True, True, True],
        }
    )
    dataframe.to_csv(predictions_path, index=False)


def test_run_temperature_scaling_writes_outputs(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "validation_predictions.csv"
    output_json = tmp_path / "temperature_scaling.json"
    output_predictions_csv = tmp_path / "validation_predictions_temperature_scaled.csv"
    _write_predictions_csv(predictions_csv)

    result = run_temperature_scaling(
        predictions_csv=predictions_csv,
        output_json=output_json,
        output_predictions_csv=output_predictions_csv,
    )

    saved_json = json.loads(output_json.read_text(encoding="utf-8"))
    scaled_predictions = pd.read_csv(output_predictions_csv)

    assert output_json.exists()
    assert output_predictions_csv.exists()
    assert result["num_examples"] == 4
    assert saved_json["num_examples"] == 4
    assert saved_json["source_predictions_csv"] == str(predictions_csv)
    assert saved_json["learned_temperature"] > 0.0
    assert "metrics_before" in saved_json
    assert "metrics_after" in saved_json
    assert "metric_deltas" in saved_json
    assert saved_json["metrics_before"]["accuracy"] == 1.0
    assert saved_json["metrics_after"]["accuracy"] == 1.0
    assert "negative_log_likelihood" in saved_json["metrics_before"]
    assert "expected_calibration_error" in saved_json["metrics_after"]
    assert "confusion_matrix" in saved_json["metrics_before"]

    assert list(scaled_predictions.columns) == [
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
        "temperature_scaled_probability_class_0",
        "temperature_scaled_probability_class_1",
        "temperature_scaled_confidence",
        "temperature_scaled_predicted_label",
        "temperature_scaled_is_correct",
    ]
    assert len(scaled_predictions) == 4
    assert (
        scaled_predictions["temperature_scaled_probability_class_0"]
        + scaled_predictions["temperature_scaled_probability_class_1"]
    ).apply(lambda value: math.isclose(value, 1.0, rel_tol=1e-6, abs_tol=1e-6)).all()
    assert (
        scaled_predictions["temperature_scaled_predicted_label"]
        == scaled_predictions["predicted_label"]
    ).all()


def test_run_temperature_scaling_rejects_missing_logits(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "validation_predictions.csv"
    output_json = tmp_path / "temperature_scaling.json"
    output_predictions_csv = tmp_path / "validation_predictions_temperature_scaled.csv"
    pd.DataFrame(
        {
            "true_label": [0, 1],
            "probability_class_0": [0.7, 0.3],
            "probability_class_1": [0.3, 0.7],
        }
    ).to_csv(predictions_csv, index=False)

    with pytest.raises(ValueError, match="missing required logit columns"):
        run_temperature_scaling(
            predictions_csv=predictions_csv,
            output_json=output_json,
            output_predictions_csv=output_predictions_csv,
        )


def test_temperature_scaling_main_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    predictions_csv = tmp_path / "validation_predictions.csv"
    output_json = tmp_path / "temperature_scaling.json"
    output_predictions_csv = tmp_path / "validation_predictions_temperature_scaled.csv"
    _write_predictions_csv(predictions_csv)

    main(
        [
            "--predictions-csv",
            str(predictions_csv),
            "--output-json",
            str(output_json),
            "--output-predictions-csv",
            str(output_predictions_csv),
        ]
    )

    captured = capsys.readouterr()
    assert "Learned temperature:" in captured.out
    assert "Saved calibration summary to:" in captured.out
    assert "Saved scaled predictions to:" in captured.out
