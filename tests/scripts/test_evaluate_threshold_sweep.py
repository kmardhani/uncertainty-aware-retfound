"""Tests for threshold-sweep analysis over saved validation prediction CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.evaluate_threshold_sweep import (
    evaluate_threshold_sweep,
    main,
    run_threshold_sweep_analysis,
)


def _write_predictions_csv(
    output_path: Path,
    *,
    positive_column: str = "probability_class_1",
) -> None:
    pd.DataFrame(
        [
            {"id_code": "a", "true_label": 0, positive_column: 0.10},
            {"id_code": "b", "true_label": 1, positive_column: 0.90},
            {"id_code": "c", "true_label": 1, positive_column: 0.40},
            {"id_code": "d", "true_label": 0, positive_column: 0.70},
        ]
    ).to_csv(output_path, index=False)


def test_evaluate_threshold_sweep_returns_expected_metrics(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    _write_predictions_csv(predictions_csv)

    result = evaluate_threshold_sweep(
        predictions_csv=predictions_csv,
        thresholds=[0.5, 0.8],
    )

    assert result["positive_probability_column"] == "probability_class_1"
    assert len(result["threshold_results"]) == 2

    threshold_05 = result["threshold_results"][0]
    threshold_08 = result["threshold_results"][1]

    assert threshold_05["threshold"] == pytest.approx(0.5)
    assert threshold_05["accuracy"] == pytest.approx(0.5)
    assert threshold_05["sensitivity"] == pytest.approx(0.5)
    assert threshold_05["specificity"] == pytest.approx(0.5)
    assert threshold_05["precision"] == pytest.approx(0.5)
    assert threshold_05["f1"] == pytest.approx(0.5)
    assert threshold_05["balanced_accuracy"] == pytest.approx(0.5)
    assert threshold_05["true_positives"] == 1
    assert threshold_05["true_negatives"] == 1
    assert threshold_05["false_positives"] == 1
    assert threshold_05["false_negatives"] == 1
    assert threshold_05["referral_rate"] == pytest.approx(0.5)
    assert threshold_05["human_review_rate"] == pytest.approx(0.5)

    assert threshold_08["threshold"] == pytest.approx(0.8)
    assert threshold_08["accuracy"] == pytest.approx(0.75)
    assert threshold_08["sensitivity"] == pytest.approx(0.5)
    assert threshold_08["specificity"] == pytest.approx(1.0)
    assert threshold_08["false_negatives"] == 1
    assert threshold_08["false_positives"] == 0
    assert threshold_08["referral_rate"] == pytest.approx(0.25)
    assert threshold_08["human_review_rate"] == pytest.approx(0.25)


def test_evaluate_threshold_sweep_uses_default_grid(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    _write_predictions_csv(predictions_csv)

    result = evaluate_threshold_sweep(predictions_csv=predictions_csv)

    assert len(result["threshold_results"]) == 99
    assert result["threshold_results"][0]["threshold"] == pytest.approx(0.01)
    assert result["threshold_results"][-1]["threshold"] == pytest.approx(0.99)


def test_evaluate_threshold_sweep_supports_explicit_probability_column(
    tmp_path: Path,
) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    _write_predictions_csv(
        predictions_csv,
        positive_column="temperature_scaled_probability_class_1",
    )

    result = evaluate_threshold_sweep(
        predictions_csv=predictions_csv,
        thresholds=[0.5],
        positive_probability_column="temperature_scaled_probability_class_1",
    )

    assert result["positive_probability_column"] == "temperature_scaled_probability_class_1"
    assert len(result["threshold_results"]) == 1


def test_run_threshold_sweep_analysis_writes_nested_outputs(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    output_json = tmp_path / "nested" / "analysis" / "threshold_sweep.json"
    output_csv = tmp_path / "nested" / "analysis" / "threshold_sweep.csv"
    _write_predictions_csv(predictions_csv)

    result = run_threshold_sweep_analysis(
        predictions_csv=predictions_csv,
        output_json=output_json,
        output_csv=output_csv,
        thresholds=[0.5, 0.8],
    )

    saved_json = json.loads(output_json.read_text(encoding="utf-8"))
    saved_csv = pd.read_csv(output_csv)

    assert output_json.exists()
    assert output_csv.exists()
    assert len(result["threshold_results"]) == 2
    assert len(saved_json["threshold_results"]) == 2
    assert len(saved_csv) == 2
    assert "human_review_rate" in saved_csv.columns


def test_main_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    output_json = tmp_path / "threshold_sweep.json"
    output_csv = tmp_path / "threshold_sweep.csv"
    _write_predictions_csv(predictions_csv)

    main(
        [
            "--predictions-csv",
            str(predictions_csv),
            "--thresholds",
            "0.5",
            "0.8",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
    )

    captured = capsys.readouterr()
    assert "Evaluated" in captured.out
    assert "Saved threshold sweep summary to:" in captured.out
    assert "Saved threshold sweep table to:" in captured.out
