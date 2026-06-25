"""Build consolidated publication-style experiment summary tables."""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def _serialize_for_json(value: Any) -> Any:
    """Convert nested outputs into JSON-serializable values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_for_json(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_serialize_for_json(item) for item in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file as a dictionary."""

    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return payload


def _warn_skip(message: str) -> None:
    """Emit a warning for a skipped optional artifact."""

    warnings.warn(message, stacklevel=2)


def _extract_confusion_counts(confusion_matrix: Any) -> tuple[int, int, int, int]:
    """Extract TN, FP, FN, TP from a 2x2 confusion matrix."""

    if (
        not isinstance(confusion_matrix, list)
        or len(confusion_matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in confusion_matrix)
    ):
        raise ValueError("Confusion matrix must be a 2x2 list.")

    tn = int(confusion_matrix[0][0])
    fp = int(confusion_matrix[0][1])
    fn = int(confusion_matrix[1][0])
    tp = int(confusion_matrix[1][1])
    return tn, fp, fn, tp


def _extract_metrics_from_payload(
    payload: dict[str, Any],
    metrics_key: str,
    source_path: Path,
) -> dict[str, Any]:
    """Extract a metrics dictionary from a JSON payload.

    Supports both explicit keys such as ``final_validation_metrics`` and
    nested evaluation-only payloads stored under ``metrics``.
    """

    metrics = payload.get(metrics_key)
    if isinstance(metrics, dict):
        return metrics

    nested_metrics = payload.get("metrics")
    if isinstance(nested_metrics, dict) and metrics_key == "metrics":
        return nested_metrics

    if isinstance(nested_metrics, dict) and "accuracy" in nested_metrics:
        return nested_metrics

    raise ValueError(f"Missing '{metrics_key}' metrics dictionary in: {source_path}")


def _build_model_row(
    *,
    model_name: str,
    source_path: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build a standardized model-comparison row."""

    explicit_fp = metrics.get("false_positives")
    explicit_fn = metrics.get("false_negatives")
    explicit_tp = metrics.get("true_positives")
    explicit_tn = metrics.get("true_negatives")

    if all(isinstance(value, (int, float)) for value in (explicit_tn, explicit_fp, explicit_fn, explicit_tp)):
        tn = int(explicit_tn)
        fp = int(explicit_fp)
        fn = int(explicit_fn)
        tp = int(explicit_tp)
    else:
        tn, fp, fn, tp = _extract_confusion_counts(metrics["confusion_matrix"])

    total = tn + fp + fn + tp
    positive_prediction_rate = (tp + fp) / total if total > 0 else None
    if positive_prediction_rate is None:
        fallback_positive_prediction_rate = metrics.get("positive_prediction_rate")
        if isinstance(fallback_positive_prediction_rate, (int, float)):
            positive_prediction_rate = float(fallback_positive_prediction_rate)

    return {
        "model": model_name,
        "source_path": str(source_path),
        "coverage": 1.0,
        "deferred_count": 0,
        "accuracy": metrics.get("accuracy"),
        "auc": metrics.get("auc"),
        "sensitivity": metrics.get("sensitivity"),
        "specificity": metrics.get("specificity"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "ece": metrics.get("expected_calibration_error"),
        "nll": metrics.get("negative_log_likelihood"),
        "brier": metrics.get("brier_score"),
        "false_positives": fp,
        "false_negatives": fn,
        "referral_rate": 0.0,
        "positive_prediction_rate": positive_prediction_rate,
    }


def build_model_comparison_table(base_output_dir: str | Path) -> pd.DataFrame:
    """Build the consolidated model-comparison table."""

    base_path = Path(base_output_dir)
    feature_heads = base_path / "feature_heads"

    model_specs: list[tuple[str, Path, str]] = [
        (
            "cached_softmax",
            feature_heads / "retfound_softmax_linear_5epoch_bs8" / "metrics.json",
            "final_validation_metrics",
        ),
        (
            "cached_softmax_temperature_scaled",
            feature_heads / "retfound_softmax_linear_5epoch_bs8" / "temperature_scaling.json",
            "metrics_after",
        ),
        (
            "variational_bayesian_val_loss_selected",
            feature_heads / "retfound_variational_bayesian_20epoch_best_val_loss" / "best_metrics.json",
            "final_validation_metrics",
        ),
        (
            "variational_bayesian_max_sensitivity_sweep",
            feature_heads / "sweeps" / "bayes_sensitivity_kl_0.001_prior_2.0" / "best_metrics.json",
            "final_validation_metrics",
        ),
        (
            "variational_bayesian_balanced_sweep",
            feature_heads / "sweeps" / "bayes_sensitivity_kl_0.00003_prior_2.0" / "best_metrics.json",
            "final_validation_metrics",
        ),
        (
            "laplace_val_loss_selected",
            feature_heads / "retfound_laplace_20epoch_best_val_loss" / "best_metrics.json",
            "final_validation_metrics",
        ),
        (
            "laplace_sensitivity_selected",
            feature_heads / "retfound_laplace_20epoch_best_sensitivity" / "best_metrics.json",
            "final_validation_metrics",
        ),
        (
            "laplace_prior_precision_best_sensitivity",
            feature_heads / "sweeps" / "laplace_sensitivity_prior_precision_0.03" / "best_metrics.json",
            "final_validation_metrics",
        ),
        (
            "laplace_prior_precision_better_calibrated",
            feature_heads / "sweeps" / "laplace_sensitivity_prior_precision_10.0" / "best_metrics.json",
            "final_validation_metrics",
        ),
        (
            "sngp_val_loss_selected",
            feature_heads / "aptos2019_retfound_sngp_val_loss" / "best_metrics.json",
            "final_validation_metrics",
        ),
        (
            "sngp_sensitivity_selected",
            feature_heads / "aptos2019_retfound_sngp_sensitivity" / "best_metrics.json",
            "final_validation_metrics",
        ),
        (
            "ddr_from_aptos_sngp_sensitivity",
            feature_heads / "ddr_from_aptos_sngp_sensitivity" / "metrics.json",
            "metrics",
        ),
        (
            "ddr_softmax",
            feature_heads / "ddr_retfound_softmax_linear_10epoch_bs16" / "metrics.json",
            "final_validation_metrics",
        ),
        (
            "ddr_bayesian_val_loss_selected",
            feature_heads
            / "ddr_retfound_variational_bayesian_20epoch_best_val_loss"
            / "best_metrics.json",
            "final_validation_metrics",
        ),
        (
            "ddr_bayesian_sensitivity_selected",
            feature_heads
            / "ddr_retfound_variational_bayesian_20epoch_best_sensitivity"
            / "best_metrics.json",
            "final_validation_metrics",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for model_name, metrics_path, metrics_key in model_specs:
        if not metrics_path.exists():
            _warn_skip(f"Skipping missing model metrics file: {metrics_path}")
            continue

        payload = _load_json(metrics_path)
        metrics = _extract_metrics_from_payload(payload, metrics_key, metrics_path)
        rows.append(
            _build_model_row(
                model_name=model_name,
                source_path=metrics_path,
                metrics=metrics,
            )
        )

    return pd.DataFrame(rows)


def build_decision_policy_table(base_output_dir: str | Path) -> pd.DataFrame:
    """Load the decision-policy comparison table."""

    path = Path(base_output_dir) / "decision_policies" / "retfound_policy_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Decision-policy comparison CSV not found: {path}")

    dataframe = pd.read_csv(path).copy()
    if "policy" not in dataframe.columns:
        raise ValueError(f"Decision-policy comparison CSV missing 'policy' column: {path}")

    if "positive_prediction_rate" not in dataframe.columns:
        if "referral_rate" in dataframe.columns:
            dataframe["positive_prediction_rate"] = dataframe["referral_rate"]
        else:
            dataframe["positive_prediction_rate"] = pd.NA

    dataframe["referral_rate"] = 0.0
    return dataframe


def _select_nearest_coverage_row(
    dataframe: pd.DataFrame,
    target_coverage: float,
) -> pd.Series:
    """Select the row with coverage closest to the requested target."""

    if "coverage" not in dataframe.columns:
        raise ValueError("Selective referral CSV must contain a 'coverage' column.")

    distance = (dataframe["coverage"] - target_coverage).abs()
    selected_index = distance.idxmin()
    return dataframe.loc[selected_index]


def _infer_selective_run_name(path: Path, selective_referral_dir: Path) -> str:
    """Infer a readable run name from a selective-referral CSV path."""

    relative_path = path.relative_to(selective_referral_dir)
    if relative_path.parent == Path("."):
        return relative_path.stem
    return relative_path.parent.name


def _infer_selective_uncertainty_name(path: Path) -> str:
    """Infer the uncertainty column name from a selective-referral CSV path."""

    stem = path.stem
    for suffix in (
        "_confidence",
        "_predictive_entropy",
        "_probability_variance",
        "_mutual_information",
        "_entropy",
        "_variance",
        "_sngp_uncertainty",
    ):
        if stem.endswith(suffix):
            return suffix.removeprefix("_")

    parent_name = path.parent.name
    for suffix in ("entropy", "variance", "predictive_entropy", "confidence", "mutual_information", "sngp_uncertainty"):
        if parent_name.endswith(f"_{suffix}"):
            return suffix

    return stem


def _iter_existing_paths(paths: Sequence[Path]) -> list[Path]:
    """Return only existing paths while preserving order and uniqueness."""

    existing_paths: list[Path] = []
    for path in paths:
        if not path.exists():
            _warn_skip(f"Skipping missing artifact: {path}")
            continue
        if path not in existing_paths:
            existing_paths.append(path)
    return existing_paths


def build_selective_referral_table(base_output_dir: str | Path) -> pd.DataFrame:
    """Build a consolidated selective-referral summary table."""

    selective_referral_dir = Path(base_output_dir) / "selective_referral"
    explicit_files = [
        selective_referral_dir / "bayes_max_sensitivity_confidence.csv",
        selective_referral_dir / "bayes_max_sensitivity_predictive_entropy.csv",
        selective_referral_dir / "bayes_max_sensitivity_probability_variance.csv",
        selective_referral_dir / "bayes_max_sensitivity_mutual_information.csv",
        selective_referral_dir
        / "aptos2019_sngp_sensitivity_entropy"
        / "selective_referral.csv",
        selective_referral_dir
        / "aptos2019_sngp_sensitivity_variance"
        / "selective_referral.csv",
        selective_referral_dir
        / "aptos2019_sngp_sensitivity_sngp_uncertainty"
        / "selective_referral.csv",
        selective_referral_dir
        / "ddr_from_aptos_sngp_sensitivity_entropy"
        / "selective_referral.csv",
        selective_referral_dir
        / "ddr_from_aptos_sngp_sensitivity_variance"
        / "selective_referral.csv",
        selective_referral_dir
        / "ddr_from_aptos_sngp_sensitivity_sngp_uncertainty"
        / "selective_referral.csv",
    ]
    discovered_ddr_files = sorted((selective_referral_dir / "ddr").glob("*.csv"))
    discovered_files = _iter_existing_paths([*explicit_files, *discovered_ddr_files])
    target_coverages = [1.0, 0.90, 0.80, 0.70]

    rows: list[dict[str, Any]] = []
    for path in discovered_files:
        dataframe = pd.read_csv(path)
        run_name = _infer_selective_run_name(path, selective_referral_dir)
        uncertainty_name = _infer_selective_uncertainty_name(path)
        for target_coverage in target_coverages:
            selected_row = _select_nearest_coverage_row(dataframe, target_coverage)
            rows.append(
                {
                    "run_name": run_name,
                    "uncertainty_column": uncertainty_name,
                    "source_path": str(path),
                    "target_coverage": float(target_coverage),
                    "coverage": float(selected_row["coverage"]),
                    "deferred_count": int(selected_row["deferred_count"]),
                    "accepted_count": int(selected_row["accepted_count"]),
                    "accuracy": float(selected_row["accuracy"]),
                    "sensitivity": float(selected_row["sensitivity"]),
                    "specificity": float(selected_row["specificity"]),
                    "balanced_accuracy": float(selected_row["balanced_accuracy"]),
                    "false_positives": int(selected_row["false_positives"]),
                    "false_negatives": int(selected_row["false_negatives"]),
                    "referral_rate": float(selected_row["referral_rate"]),
                }
            )

    return pd.DataFrame(rows)


def _select_best_balanced_accuracy_row(dataframe: pd.DataFrame) -> pd.Series:
    """Select the threshold row with the highest balanced accuracy."""

    sorted_dataframe = dataframe.sort_values(
        by=["balanced_accuracy", "specificity", "accuracy", "threshold"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return sorted_dataframe.iloc[0]


def _select_lowest_false_negatives_row(dataframe: pd.DataFrame) -> pd.Series:
    """Select the threshold row with the fewest false negatives.

    Ties prefer higher specificity, then higher accuracy, then lower threshold.
    """

    sorted_dataframe = dataframe.sort_values(
        by=["false_negatives", "specificity", "accuracy", "threshold"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    return sorted_dataframe.iloc[0]


def _build_threshold_summary_row(
    *,
    sweep_name: str,
    source_path: Path,
    selected_policy: str,
    row: pd.Series,
) -> dict[str, Any]:
    """Build a standardized threshold-sweep summary row."""

    return {
        "sweep_name": sweep_name,
        "source_path": str(source_path),
        "selected_policy": selected_policy,
        "threshold": float(row["threshold"]),
        "accuracy": float(row["accuracy"]),
        "sensitivity": float(row["sensitivity"]),
        "specificity": float(row["specificity"]),
        "balanced_accuracy": float(row["balanced_accuracy"]),
        "false_negatives": int(row["false_negatives"]),
        "false_positives": int(row["false_positives"]),
    }


def build_threshold_sweep_table(base_output_dir: str | Path) -> pd.DataFrame:
    """Build a consolidated threshold-sweep summary table."""

    threshold_sweeps_dir = Path(base_output_dir) / "threshold_sweeps"
    explicit_paths = [
        threshold_sweeps_dir / "aptos2019_sngp_sensitivity" / "threshold_sweep.csv",
        threshold_sweeps_dir / "ddr_from_aptos_sngp_sensitivity" / "threshold_sweep.csv",
    ]
    discovered_ddr_paths = sorted((threshold_sweeps_dir / "ddr").glob("*.csv"))
    sweep_paths = _iter_existing_paths([*explicit_paths, *discovered_ddr_paths])

    rows: list[dict[str, Any]] = []
    for path in sweep_paths:
        dataframe = pd.read_csv(path)
        sweep_name = (
            path.parent.name
            if path.parent != threshold_sweeps_dir / "ddr"
            else f"ddr_{path.stem}"
        )
        best_balanced_accuracy_row = _select_best_balanced_accuracy_row(dataframe)
        lowest_false_negative_row = _select_lowest_false_negatives_row(dataframe)
        rows.append(
            _build_threshold_summary_row(
                sweep_name=sweep_name,
                source_path=path,
                selected_policy="best_balanced_accuracy",
                row=best_balanced_accuracy_row,
            )
        )
        rows.append(
            _build_threshold_summary_row(
                sweep_name=sweep_name,
                source_path=path,
                selected_policy="lowest_false_negatives",
                row=lowest_false_negative_row,
            )
        )

    return pd.DataFrame(rows)


def run_build_experiment_summary_tables(
    base_output_dir: str | Path,
    summary_output_dir: str | Path,
) -> dict[str, Any]:
    """Build and write consolidated experiment summary tables."""

    summary_dir = Path(summary_output_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    model_comparison = build_model_comparison_table(base_output_dir)
    decision_policy_comparison = build_decision_policy_table(base_output_dir)
    selective_referral_summary = build_selective_referral_table(base_output_dir)
    threshold_sweep_summary = build_threshold_sweep_table(base_output_dir)

    model_csv_path = summary_dir / "model_comparison.csv"
    decision_csv_path = summary_dir / "decision_policy_comparison.csv"
    selective_csv_path = summary_dir / "selective_referral_summary.csv"
    threshold_csv_path = summary_dir / "threshold_sweep_summary.csv"
    json_path = summary_dir / "summary_tables.json"

    model_comparison.to_csv(model_csv_path, index=False)
    decision_policy_comparison.to_csv(decision_csv_path, index=False)
    selective_referral_summary.to_csv(selective_csv_path, index=False)
    threshold_sweep_summary.to_csv(threshold_csv_path, index=False)

    payload = {
        "base_output_dir": str(base_output_dir),
        "model_comparison": model_comparison.to_dict(orient="records"),
        "decision_policy_comparison": decision_policy_comparison.to_dict(orient="records"),
        "selective_referral_summary": selective_referral_summary.to_dict(orient="records"),
        "threshold_sweep_summary": threshold_sweep_summary.to_dict(orient="records"),
    }
    json_path.write_text(
        json.dumps(_serialize_for_json(payload), indent=2),
        encoding="utf-8",
    )

    return {
        "model_comparison_csv": str(model_csv_path),
        "decision_policy_comparison_csv": str(decision_csv_path),
        "selective_referral_summary_csv": str(selective_csv_path),
        "threshold_sweep_summary_csv": str(threshold_csv_path),
        "summary_tables_json": str(json_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for summary-table building."""

    parser = argparse.ArgumentParser(
        description="Build consolidated experiment comparison summary tables."
    )
    parser.add_argument("--base-output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--summary-output-dir",
        type=Path,
        default=Path("outputs") / "summary_tables",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the summary-table builder CLI."""

    args = parse_args(argv)
    result = run_build_experiment_summary_tables(
        base_output_dir=args.base_output_dir,
        summary_output_dir=args.summary_output_dir,
    )
    print(f"Saved model comparison to: {result['model_comparison_csv']}")
    print(f"Saved decision-policy comparison to: {result['decision_policy_comparison_csv']}")
    print(f"Saved selective-referral summary to: {result['selective_referral_summary_csv']}")
    print(f"Saved threshold-sweep summary to: {result['threshold_sweep_summary_csv']}")
    print(f"Saved JSON summary to: {result['summary_tables_json']}")


if __name__ == "__main__":
    main()
