"""Tests for selective referral analysis over saved validation prediction CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.evaluate_selective_referral import (
    evaluate_selective_referral,
    main,
    run_selective_referral_analysis,
)


def _write_predictions_csv(output_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "id_code": "a",
                "true_label": 0,
                "predicted_label": 0,
                "confidence": 0.95,
                "predictive_entropy": 0.05,
            },
            {
                "id_code": "b",
                "true_label": 1,
                "predicted_label": 1,
                "confidence": 0.90,
                "predictive_entropy": 0.10,
            },
            {
                "id_code": "c",
                "true_label": 1,
                "predicted_label": 0,
                "confidence": 0.55,
                "predictive_entropy": 0.60,
            },
            {
                "id_code": "d",
                "true_label": 0,
                "predicted_label": 1,
                "confidence": 0.60,
                "predictive_entropy": 0.55,
            },
            {
                "id_code": "e",
                "true_label": 1,
                "predicted_label": 1,
                "confidence": 0.85,
                "predictive_entropy": 0.20,
            },
        ]
    ).to_csv(output_path, index=False)


def test_evaluate_selective_referral_higher_is_more_uncertain(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    _write_predictions_csv(predictions_csv)

    results = evaluate_selective_referral(
        predictions_csv=predictions_csv,
        uncertainty_column="predictive_entropy",
        higher_is_more_uncertain=True,
        coverage_levels=[1.0, 0.8],
    )

    assert len(results) == 2
    full_result = results[0]
    partial_result = results[1]

    assert full_result["coverage"] == pytest.approx(1.0)
    assert full_result["target_coverage"] == pytest.approx(1.0)
    assert full_result["total_count"] == 5
    assert full_result["accepted_count"] == 5
    assert full_result["deferred_count"] == 0
    assert full_result["referral_rate"] == pytest.approx(0.0)
    assert full_result["coverage"] == pytest.approx(full_result["accepted_count"] / 5)
    assert full_result["false_negatives"] == 1
    assert full_result["false_positives"] == 1
    assert full_result["accepted_positive_prediction_rate"] == pytest.approx(0.6)

    assert partial_result["coverage"] == pytest.approx(0.8)
    assert partial_result["target_coverage"] == pytest.approx(0.8)
    assert partial_result["total_count"] == 5
    assert partial_result["accepted_count"] == 4
    assert partial_result["deferred_count"] == 1
    assert partial_result["referral_rate"] == pytest.approx(0.2)
    assert partial_result["coverage"] == pytest.approx(partial_result["accepted_count"] / 5)
    assert partial_result["false_negatives"] == 0
    assert partial_result["false_positives"] == 1
    assert partial_result["accepted_positive_prediction_rate"] == pytest.approx(0.75)


def test_evaluate_selective_referral_lower_is_more_uncertain(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    _write_predictions_csv(predictions_csv)

    results = evaluate_selective_referral(
        predictions_csv=predictions_csv,
        uncertainty_column="confidence",
        higher_is_more_uncertain=False,
        coverage_levels=[0.6],
    )

    result = results[0]
    assert result["coverage"] == pytest.approx(0.6)
    assert result["target_coverage"] == pytest.approx(0.6)
    assert result["total_count"] == 5
    assert result["accepted_count"] == 3
    assert result["deferred_count"] == 2
    assert result["referral_rate"] == pytest.approx(0.4)
    assert result["coverage"] == pytest.approx(result["accepted_count"] / 5)
    assert result["false_negatives"] == 0


def test_run_selective_referral_analysis_writes_nested_outputs(tmp_path: Path) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    output_json = tmp_path / "nested" / "analysis" / "selective_referral.json"
    output_csv = tmp_path / "nested" / "analysis" / "selective_referral.csv"
    _write_predictions_csv(predictions_csv)

    result = run_selective_referral_analysis(
        predictions_csv=predictions_csv,
        uncertainty_column="predictive_entropy",
        higher_is_more_uncertain=True,
        coverage_levels=[1.0, 0.8, 0.6],
        output_json=output_json,
        output_csv=output_csv,
    )

    saved_json = json.loads(output_json.read_text(encoding="utf-8"))
    saved_csv = pd.read_csv(output_csv)

    assert output_json.exists()
    assert output_csv.exists()
    assert len(result["coverage_results"]) == 3
    assert len(saved_json["coverage_results"]) == 3
    assert len(saved_csv) == 3


def test_main_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    predictions_csv = tmp_path / "predictions.csv"
    output_json = tmp_path / "selective_referral.json"
    output_csv = tmp_path / "selective_referral.csv"
    _write_predictions_csv(predictions_csv)

    main(
        [
            "--predictions-csv",
            str(predictions_csv),
            "--uncertainty-column",
            "predictive_entropy",
            "--higher-is-more-uncertain",
            "--coverage-levels",
            "1.0",
            "0.8",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
    )

    captured = capsys.readouterr()
    assert "Evaluated" in captured.out
    assert "Saved selective referral summary to:" in captured.out
    assert "Saved selective referral table to:" in captured.out
