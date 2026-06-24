"""Tests for consolidated experiment summary-table generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analysis.build_experiment_summary_tables import (
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

    decision_policy_dir = base_output_dir / "decision_policies"
    decision_policy_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"policy": "model:softmax_temp", "accuracy": 0.88, "false_negatives": 5},
            {"policy": "policy:or_rule", "accuracy": 0.89, "false_negatives": 4},
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
                "referral_rate": 0.40,
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
                "referral_rate": 0.39,
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
                "referral_rate": 0.38,
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
                "referral_rate": 0.35,
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
    summary_json = json.loads((summary_output_dir / "summary_tables.json").read_text(encoding="utf-8"))

    assert Path(result["model_comparison_csv"]).exists()
    assert Path(result["decision_policy_comparison_csv"]).exists()
    assert Path(result["selective_referral_summary_csv"]).exists()
    assert Path(result["summary_tables_json"]).exists()

    assert len(model_comparison) == 9
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
    }
    assert len(decision_policy) == 2
    assert len(selective_referral) == 16
    assert set(selective_referral["target_coverage"]) == {1.0, 0.9, 0.8, 0.7}
    assert "model_comparison" in summary_json
    assert "decision_policy_comparison" in summary_json
    assert "selective_referral_summary" in summary_json


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
    assert "Saved JSON summary to:" in captured.out
