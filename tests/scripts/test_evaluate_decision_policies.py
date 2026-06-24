"""Tests for decision-policy analysis over saved validation prediction CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.evaluate_decision_policies import (
    evaluate_decision_policies,
    main,
    merge_prediction_csvs,
    run_policy_analysis,
)


def _write_predictions_csv(
    output_path: Path,
    *,
    rows: list[dict[str, object]],
) -> None:
    pd.DataFrame(rows).to_csv(output_path, index=False)


def test_evaluate_decision_policies_writes_expected_outputs(tmp_path: Path) -> None:
    model_a_path = tmp_path / "model_a.csv"
    model_b_path = tmp_path / "model_b.csv"
    output_json = tmp_path / "policy_results.json"
    output_csv = tmp_path / "policy_results.csv"

    _write_predictions_csv(
        model_a_path,
        rows=[
            {"id_code": "a", "true_label": 0, "predicted_label": 0, "image_path": "/tmp/a.png"},
            {"id_code": "b", "true_label": 1, "predicted_label": 1, "image_path": "/tmp/b.png"},
            {"id_code": "c", "true_label": 1, "predicted_label": 0, "image_path": "/tmp/c.png"},
            {"id_code": "d", "true_label": 0, "predicted_label": 0, "image_path": "/tmp/d.png"},
        ],
    )
    _write_predictions_csv(
        model_b_path,
        rows=[
            {"id_code": "a", "true_label": 0, "predicted_label": 1, "image_path": "/tmp/a.png"},
            {"id_code": "b", "true_label": 1, "predicted_label": 1, "image_path": "/tmp/b.png"},
            {"id_code": "c", "true_label": 1, "predicted_label": 1, "image_path": "/tmp/c.png"},
            {"id_code": "d", "true_label": 0, "predicted_label": 0, "image_path": "/tmp/d.png"},
        ],
    )

    result = run_policy_analysis(
        prediction_pairs=[
            f"model_a={model_a_path}",
            f"model_b={model_b_path}",
        ],
        output_json=output_json,
        output_csv=output_csv,
    )

    saved_json = json.loads(output_json.read_text(encoding="utf-8"))
    saved_csv = pd.read_csv(output_csv)

    assert output_json.exists()
    assert output_csv.exists()
    assert len(result["policy_results"]) == 5
    assert len(saved_json["policy_results"]) == 5
    assert len(saved_csv) == 5
    assert set(saved_csv["policy"]) == {
        "model:model_a",
        "model:model_b",
        "policy:or_rule",
        "policy:majority_vote",
        "policy:and_rule",
    }

    model_a_result = saved_csv.loc[saved_csv["policy"] == "model:model_a"].iloc[0]
    model_b_result = saved_csv.loc[saved_csv["policy"] == "model:model_b"].iloc[0]
    or_result = saved_csv.loc[saved_csv["policy"] == "policy:or_rule"].iloc[0]
    majority_result = saved_csv.loc[saved_csv["policy"] == "policy:majority_vote"].iloc[0]
    and_result = saved_csv.loc[saved_csv["policy"] == "policy:and_rule"].iloc[0]

    assert model_a_result["accuracy"] == pytest.approx(0.75)
    assert model_a_result["sensitivity"] == pytest.approx(0.5)
    assert model_a_result["specificity"] == pytest.approx(1.0)
    assert model_a_result["false_negatives"] == 1

    assert model_b_result["accuracy"] == pytest.approx(0.75)
    assert model_b_result["sensitivity"] == pytest.approx(1.0)
    assert model_b_result["specificity"] == pytest.approx(0.5)
    assert model_b_result["false_positives"] == 1

    assert or_result["sensitivity"] == pytest.approx(1.0)
    assert or_result["false_negatives"] == 0
    assert majority_result["sensitivity"] == pytest.approx(1.0)
    assert majority_result["false_negatives"] == 0
    assert and_result["sensitivity"] == pytest.approx(0.5)
    assert and_result["specificity"] == pytest.approx(1.0)
    assert and_result["false_negatives"] == 1


def test_merge_prediction_csvs_requires_matching_true_labels(tmp_path: Path) -> None:
    model_a_path = tmp_path / "model_a.csv"
    model_b_path = tmp_path / "model_b.csv"

    _write_predictions_csv(
        model_a_path,
        rows=[
            {"id_code": "a", "true_label": 0, "predicted_label": 0},
            {"id_code": "b", "true_label": 1, "predicted_label": 1},
        ],
    )
    _write_predictions_csv(
        model_b_path,
        rows=[
            {"id_code": "a", "true_label": 1, "predicted_label": 0},
            {"id_code": "b", "true_label": 1, "predicted_label": 1},
        ],
    )

    with pytest.raises(ValueError, match="mismatched true_label"):
        merge_prediction_csvs(
            {
                "model_a": model_a_path,
                "model_b": model_b_path,
            }
        )


def test_evaluate_decision_policies_returns_policy_metrics(tmp_path: Path) -> None:
    model_a_path = tmp_path / "model_a.csv"

    _write_predictions_csv(
        model_a_path,
        rows=[
            {"id_code": "a", "true_label": 0, "predicted_label": 0},
            {"id_code": "b", "true_label": 1, "predicted_label": 1},
        ],
    )

    policy_results = evaluate_decision_policies({"model_a": model_a_path})

    assert len(policy_results) == 4
    assert {result["policy"] for result in policy_results} == {
        "model:model_a",
        "policy:or_rule",
        "policy:majority_vote",
        "policy:and_rule",
    }


def test_main_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_a_path = tmp_path / "model_a.csv"
    model_b_path = tmp_path / "model_b.csv"
    output_json = tmp_path / "policy_results.json"
    output_csv = tmp_path / "policy_results.csv"

    _write_predictions_csv(
        model_a_path,
        rows=[
            {"id_code": "a", "true_label": 0, "predicted_label": 0},
            {"id_code": "b", "true_label": 1, "predicted_label": 1},
        ],
    )
    _write_predictions_csv(
        model_b_path,
        rows=[
            {"id_code": "a", "true_label": 0, "predicted_label": 1},
            {"id_code": "b", "true_label": 1, "predicted_label": 1},
        ],
    )

    main(
        [
            "--predictions",
            f"model_a={model_a_path}",
            "--predictions",
            f"model_b={model_b_path}",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
    )

    captured = capsys.readouterr()
    assert "Evaluated" in captured.out
    assert "Saved policy summary to:" in captured.out
    assert "Saved policy table to:" in captured.out
