"""Tests for consolidated experiment summary-table generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.build_experiment_summary_tables import (
    build_selective_referral_table,
    build_model_comparison_table,
    main,
    run_build_experiment_summary_tables,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _metrics_payload(
    *,
    accuracy: float,
    auc: float,
    sensitivity: float,
    specificity: float,
    balanced_accuracy: float,
    ece: float,
    nll: float,
    brier: float,
    confusion_matrix: list[list[int]],
) -> dict[str, object]:
    return {
        "accuracy": accuracy,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "expected_calibration_error": ece,
        "negative_log_likelihood": nll,
        "brier_score": brier,
        "confusion_matrix": confusion_matrix,
    }


def _write_synthetic_outputs(base_output_dir: Path) -> None:
    feature_heads = base_output_dir / "feature_heads"
    model_specs = [
        ("retfound_softmax_linear_5epoch_bs8/metrics.json", "final_validation_metrics", 0.87),
        ("retfound_softmax_linear_5epoch_bs8/temperature_scaling.json", "metrics_after", 0.88),
        ("retfound_variational_bayesian_20epoch_best_val_loss/best_metrics.json", "final_validation_metrics", 0.89),
        ("sweeps/bayes_sensitivity_kl_0.001_prior_2.0/best_metrics.json", "final_validation_metrics", 0.90),
        ("sweeps/bayes_sensitivity_kl_0.00003_prior_2.0/best_metrics.json", "final_validation_metrics", 0.91),
        ("retfound_laplace_20epoch_best_val_loss/best_metrics.json", "final_validation_metrics", 0.86),
        ("retfound_laplace_20epoch_best_sensitivity/best_metrics.json", "final_validation_metrics", 0.85),
        ("sweeps/laplace_sensitivity_prior_precision_0.03/best_metrics.json", "final_validation_metrics", 0.84),
        ("sweeps/laplace_sensitivity_prior_precision_10.0/best_metrics.json", "final_validation_metrics", 0.83),
        ("aptos2019_retfound_sngp_val_loss/best_metrics.json", "final_validation_metrics", 0.88),
        ("aptos2019_retfound_sngp_sensitivity/best_metrics.json", "final_validation_metrics", 0.87),
        ("ddr_retfound_softmax_linear_10epoch_bs16/metrics.json", "final_validation_metrics", 0.79),
        (
            "ddr_retfound_variational_bayesian_20epoch_best_val_loss/best_metrics.json",
            "final_validation_metrics",
            0.80,
        ),
        (
            "ddr_retfound_variational_bayesian_20epoch_best_sensitivity/best_metrics.json",
            "final_validation_metrics",
            0.77,
        ),
    ]

    for relative_path, metrics_key, accuracy in model_specs:
        _write_json(
            feature_heads / relative_path,
            {
                metrics_key: _metrics_payload(
                    accuracy=accuracy,
                    auc=0.95,
                    sensitivity=0.94,
                    specificity=0.86,
                    balanced_accuracy=0.90,
                    ece=0.03,
                    nll=0.25,
                    brier=0.08,
                    confusion_matrix=[[18, 2], [3, 17]],
                )
            },
        )

    _write_json(
        feature_heads / "ddr_from_aptos_sngp_sensitivity" / "metrics.json",
        {
            "metrics": _metrics_payload(
                accuracy=0.75,
                auc=0.86,
                sensitivity=0.84,
                specificity=0.70,
                balanced_accuracy=0.77,
                ece=0.02,
                nll=0.47,
                brier=0.15,
                confusion_matrix=[[14, 6], [3, 17]],
            )
        },
    )

    decision_policy_dir = base_output_dir / "decision_policies"
    decision_policy_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "policy": "model:softmax_temp",
                "accuracy": 0.88,
                "false_negatives": 5,
                "referral_rate": 0.45,
            },
            {
                "policy": "policy:or_rule",
                "accuracy": 0.89,
                "false_negatives": 4,
                "referral_rate": 0.55,
            },
        ]
    ).to_csv(decision_policy_dir / "retfound_policy_comparison.csv", index=False)

    selective_referral_dir = base_output_dir / "selective_referral"
    selective_referral_dir.mkdir(parents=True, exist_ok=True)
    selective_template = pd.DataFrame(
        [
            {
                "coverage": 1.0,
                "deferred_count": 0,
                "accepted_count": 10,
                "accuracy": 0.80,
                "sensitivity": 0.90,
                "specificity": 0.70,
                "balanced_accuracy": 0.80,
                "false_positives": 2,
                "false_negatives": 1,
                "referral_rate": 0.5,
            },
            {
                "coverage": 0.91,
                "deferred_count": 1,
                "accepted_count": 9,
                "accuracy": 0.82,
                "sensitivity": 0.92,
                "specificity": 0.72,
                "balanced_accuracy": 0.82,
                "false_positives": 2,
                "false_negatives": 1,
                "referral_rate": 0.1,
            },
            {
                "coverage": 0.81,
                "deferred_count": 2,
                "accepted_count": 8,
                "accuracy": 0.85,
                "sensitivity": 0.95,
                "specificity": 0.75,
                "balanced_accuracy": 0.85,
                "false_positives": 1,
                "false_negatives": 0,
                "referral_rate": 0.2,
            },
            {
                "coverage": 0.69,
                "deferred_count": 3,
                "accepted_count": 7,
                "accuracy": 0.87,
                "sensitivity": 0.97,
                "specificity": 0.78,
                "balanced_accuracy": 0.88,
                "false_positives": 1,
                "false_negatives": 0,
                "referral_rate": 0.3,
            },
        ]
    )
    for uncertainty_name in (
        "confidence",
        "predictive_entropy",
        "probability_variance",
        "mutual_information",
    ):
        selective_template.to_csv(
            selective_referral_dir / f"bayes_max_sensitivity_{uncertainty_name}.csv",
            index=False,
        )

    for relative_path in (
        "aptos2019_sngp_sensitivity_entropy/selective_referral.csv",
        "aptos2019_sngp_sensitivity_variance/selective_referral.csv",
        "aptos2019_sngp_sensitivity_sngp_uncertainty/selective_referral.csv",
        "ddr_from_aptos_sngp_sensitivity_entropy/selective_referral.csv",
        "ddr_from_aptos_sngp_sensitivity_variance/selective_referral.csv",
        "ddr_from_aptos_sngp_sensitivity_sngp_uncertainty/selective_referral.csv",
        "ddr/bayesian_confidence.csv",
        "ddr/bayesian_predictive_entropy.csv",
    ):
        output_path = selective_referral_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        selective_template.to_csv(output_path, index=False)

    threshold_sweeps_dir = base_output_dir / "threshold_sweeps"
    threshold_template = pd.DataFrame(
        [
            {
                "threshold": 0.10,
                "accuracy": 0.75,
                "sensitivity": 0.95,
                "specificity": 0.60,
                "balanced_accuracy": 0.775,
                "false_negatives": 2,
                "false_positives": 6,
            },
            {
                "threshold": 0.30,
                "accuracy": 0.82,
                "sensitivity": 0.88,
                "specificity": 0.80,
                "balanced_accuracy": 0.84,
                "false_negatives": 3,
                "false_positives": 2,
            },
            {
                "threshold": 0.50,
                "accuracy": 0.79,
                "sensitivity": 0.80,
                "specificity": 0.86,
                "balanced_accuracy": 0.83,
                "false_negatives": 1,
                "false_positives": 4,
            },
        ]
    )
    for relative_path in (
        "aptos2019_sngp_sensitivity/threshold_sweep.csv",
        "ddr_from_aptos_sngp_sensitivity/threshold_sweep.csv",
        "ddr/softmax_temp_threshold_sweep.csv",
        "ddr/bayesian_sensitivity_threshold_sweep.csv",
    ):
        output_path = threshold_sweeps_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        threshold_template.to_csv(output_path, index=False)


def test_run_build_experiment_summary_tables_writes_expected_outputs(tmp_path: Path) -> None:
    base_output_dir = tmp_path / "outputs"
    summary_output_dir = base_output_dir / "summary_tables"
    _write_synthetic_outputs(base_output_dir)

    result = run_build_experiment_summary_tables(
        base_output_dir=base_output_dir,
        summary_output_dir=summary_output_dir,
    )

    model_comparison = pd.read_csv(summary_output_dir / "model_comparison.csv")
    decision_policy = pd.read_csv(summary_output_dir / "decision_policy_comparison.csv")
    selective_referral = pd.read_csv(summary_output_dir / "selective_referral_summary.csv")
    threshold_sweep = pd.read_csv(summary_output_dir / "threshold_sweep_summary.csv")
    summary_json = json.loads((summary_output_dir / "summary_tables.json").read_text(encoding="utf-8"))

    assert Path(result["model_comparison_csv"]).exists()
    assert Path(result["decision_policy_comparison_csv"]).exists()
    assert Path(result["selective_referral_summary_csv"]).exists()
    assert Path(result["threshold_sweep_summary_csv"]).exists()
    assert Path(result["summary_tables_json"]).exists()

    assert len(model_comparison) == 15
    assert set(model_comparison["model"]) == {
        "cached_softmax",
        "cached_softmax_temperature_scaled",
        "variational_bayesian_val_loss_selected",
        "variational_bayesian_max_sensitivity_sweep",
        "variational_bayesian_balanced_sweep",
        "laplace_val_loss_selected",
        "laplace_sensitivity_selected",
        "laplace_prior_precision_best_sensitivity",
        "laplace_prior_precision_better_calibrated",
        "sngp_val_loss_selected",
        "sngp_sensitivity_selected",
        "ddr_from_aptos_sngp_sensitivity",
        "ddr_softmax",
        "ddr_bayesian_val_loss_selected",
        "ddr_bayesian_sensitivity_selected",
    }
    assert model_comparison["coverage"].eq(1.0).all()
    assert model_comparison["deferred_count"].eq(0).all()
    assert model_comparison["referral_rate"].eq(0.0).all()
    assert "positive_prediction_rate" in model_comparison.columns
    assert model_comparison["positive_prediction_rate"].notna().all()
    expected_positive_prediction_rates = {
        "ddr_from_aptos_sngp_sensitivity": (17 + 6) / 40,
    }
    for model_name in model_comparison["model"]:
        expected_positive_prediction_rates.setdefault(model_name, (17 + 2) / 40)
    for row in model_comparison.itertuples(index=False):
        assert row.positive_prediction_rate == pytest.approx(
            expected_positive_prediction_rates[row.model]
        )
    assert len(decision_policy) == 2
    assert decision_policy["referral_rate"].eq(0.0).all()
    assert "positive_prediction_rate" in decision_policy.columns
    assert decision_policy["positive_prediction_rate"].tolist() == pytest.approx([0.45, 0.55])
    assert len(selective_referral) == 48
    assert set(selective_referral["target_coverage"]) == {1.0, 0.9, 0.8, 0.7}
    assert {"run_name", "uncertainty_column"}.issubset(selective_referral.columns)
    assert selective_referral["referral_rate"].eq(
        selective_referral["deferred_count"] / (
            selective_referral["accepted_count"] + selective_referral["deferred_count"]
        )
    ).all()
    assert len(threshold_sweep) == 8
    assert set(threshold_sweep["selected_policy"]) == {
        "best_balanced_accuracy",
        "lowest_false_negatives",
    }
    assert "model_comparison" in summary_json
    assert "decision_policy_comparison" in summary_json
    assert "selective_referral_summary" in summary_json
    assert "threshold_sweep_summary" in summary_json


def test_build_model_comparison_table_supports_nested_metrics_and_skips_missing(
    tmp_path: Path,
    recwarn: pytest.WarningsRecorder,
) -> None:
    base_output_dir = tmp_path / "outputs"
    feature_heads = base_output_dir / "feature_heads"

    _write_json(
        feature_heads / "ddr_from_aptos_sngp_sensitivity" / "metrics.json",
        {
            "metrics": _metrics_payload(
                accuracy=0.75,
                auc=0.86,
                sensitivity=0.84,
                specificity=0.70,
                balanced_accuracy=0.77,
                ece=0.02,
                nll=0.47,
                brier=0.15,
                confusion_matrix=[[14, 6], [3, 17]],
            )
        },
    )

    dataframe = build_model_comparison_table(base_output_dir)

    assert len(dataframe) == 1
    assert dataframe.iloc[0]["model"] == "ddr_from_aptos_sngp_sensitivity"
    assert dataframe.iloc[0]["false_positives"] == 6
    assert dataframe.iloc[0]["false_negatives"] == 3
    assert dataframe.iloc[0]["referral_rate"] == 0.0
    assert dataframe.iloc[0]["positive_prediction_rate"] == pytest.approx((17 + 6) / 40)
    warning_messages = [str(warning.message) for warning in recwarn]
    assert any("Skipping missing model metrics file" in message for message in warning_messages)


def test_build_selective_referral_table_recomputes_referral_rate_from_counts(
    tmp_path: Path,
    recwarn: pytest.WarningsRecorder,
) -> None:
    base_output_dir = tmp_path / "outputs"
    selective_referral_dir = base_output_dir / "selective_referral"
    selective_referral_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "coverage": 1.0,
                "deferred_count": 0,
                "accepted_count": 100,
                "accuracy": 0.80,
                "sensitivity": 0.90,
                "specificity": 0.70,
                "balanced_accuracy": 0.80,
                "false_positives": 2,
                "false_negatives": 1,
                "referral_rate": 0.5,
            },
            {
                "coverage": 0.8,
                "deferred_count": 20,
                "accepted_count": 80,
                "accuracy": 0.85,
                "sensitivity": 0.92,
                "specificity": 0.78,
                "balanced_accuracy": 0.85,
                "false_positives": 1,
                "false_negatives": 0,
                "referral_rate": 0.9,
            },
            {
                "coverage": 0.7,
                "deferred_count": 30,
                "accepted_count": 70,
                "total_count": 100,
                "accuracy": 0.87,
                "sensitivity": 0.95,
                "specificity": 0.80,
                "balanced_accuracy": 0.88,
                "false_positives": 1,
                "false_negatives": 0,
                "referral_rate": 0.1,
            },
        ]
    ).to_csv(selective_referral_dir / "bayes_max_sensitivity_confidence.csv", index=False)

    dataframe = build_selective_referral_table(base_output_dir)

    coverage_to_referral_rate = {
        row.coverage: row.referral_rate for row in dataframe.itertuples(index=False)
    }
    assert coverage_to_referral_rate[1.0] == pytest.approx(0.0)
    assert coverage_to_referral_rate[0.8] == pytest.approx(0.2)
    assert coverage_to_referral_rate[0.7] == pytest.approx(0.3)
    warning_messages = [str(warning.message) for warning in recwarn]
    assert any("Skipping missing artifact" in message for message in warning_messages)


def test_main_prints_summary_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base_output_dir = tmp_path / "outputs"
    summary_output_dir = base_output_dir / "summary_tables"
    _write_synthetic_outputs(base_output_dir)

    main(
        [
            "--base-output-dir",
            str(base_output_dir),
            "--summary-output-dir",
            str(summary_output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert "Saved model comparison to:" in captured.out
    assert "Saved decision-policy comparison to:" in captured.out
    assert "Saved selective-referral summary to:" in captured.out
    assert "Saved threshold-sweep summary to:" in captured.out
    assert "Saved JSON summary to:" in captured.out
