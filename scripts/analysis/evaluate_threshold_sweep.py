"""Evaluate screening operating points across positive-class probability thresholds."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from uncertainty_retfound.evaluation.metrics import (
    accuracy_score,
    balanced_accuracy_score_binary,
    f1_score_binary,
    precision_score_binary,
    recall_score_binary,
    specificity_score_binary,
)


_REQUIRED_COLUMNS: tuple[str, ...] = ("true_label",)
_DEFAULT_PROBABILITY_COLUMNS: tuple[str, ...] = (
    "probability_class_1",
    "temperature_scaled_probability_class_1",
    "positive_probability",
)


def _serialize_for_json(value: Any) -> Any:
    """Convert nested outputs into JSON-serializable values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_for_json(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_serialize_for_json(item) for item in value]
    return value


def _default_threshold_grid() -> list[float]:
    """Return the default threshold grid from 0.01 to 0.99."""

    return [round(index / 100.0, 2) for index in range(1, 100)]


def _validate_thresholds(thresholds: Sequence[float] | None) -> list[float]:
    """Validate and normalize threshold values."""

    if thresholds is None:
        return _default_threshold_grid()

    if not thresholds:
        raise ValueError("At least one threshold is required.")

    normalized_thresholds: list[float] = []
    for threshold in thresholds:
        normalized_threshold = float(threshold)
        if normalized_threshold < 0.0 or normalized_threshold > 1.0:
            raise ValueError(
                f"Thresholds must be in the interval [0, 1]. Got: {normalized_threshold}"
            )
        normalized_thresholds.append(normalized_threshold)

    return normalized_thresholds


def _resolve_positive_probability_column(
    dataframe: pd.DataFrame,
    positive_probability_column: str | None,
) -> str:
    """Resolve the positive-class probability column name."""

    if positive_probability_column is not None:
        if positive_probability_column not in dataframe.columns:
            raise ValueError(
                "Predictions CSV is missing the requested positive probability column: "
                f"{positive_probability_column}"
            )
        return positive_probability_column

    for candidate_column in _DEFAULT_PROBABILITY_COLUMNS:
        if candidate_column in dataframe.columns:
            return candidate_column

    supported_columns = ", ".join(_DEFAULT_PROBABILITY_COLUMNS)
    raise ValueError(
        "Could not identify a positive-class probability column. "
        f"Expected one of: {supported_columns}"
    )


def _load_predictions_csv(
    predictions_csv: str | Path,
    positive_probability_column: str | None,
) -> tuple[pd.DataFrame, str]:
    """Load a saved validation prediction CSV and resolve the probability column."""

    predictions_path = Path(predictions_csv)
    dataframe = pd.read_csv(predictions_path)
    missing_columns = [column for column in _REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"Predictions CSV is missing required columns: {missing}. File: {predictions_path}"
        )

    resolved_probability_column = _resolve_positive_probability_column(
        dataframe=dataframe,
        positive_probability_column=positive_probability_column,
    )
    return dataframe.copy(), resolved_probability_column


def _compute_threshold_metrics(
    labels: torch.Tensor,
    positive_probabilities: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    """Compute screening metrics for one probability threshold."""

    predictions = (positive_probabilities >= threshold).to(dtype=torch.int64)

    true_positive = int(((predictions == 1) & (labels == 1)).sum().item())
    true_negative = int(((predictions == 0) & (labels == 0)).sum().item())
    false_positive = int(((predictions == 1) & (labels == 0)).sum().item())
    false_negative = int(((predictions == 0) & (labels == 1)).sum().item())
    referral_rate = float((predictions == 1).float().mean().item())

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(predictions, labels),
        "sensitivity": recall_score_binary(labels, predictions),
        "specificity": specificity_score_binary(labels, predictions),
        "precision": precision_score_binary(labels, predictions),
        "f1": f1_score_binary(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score_binary(labels, predictions),
        "true_positives": true_positive,
        "true_negatives": true_negative,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "referral_rate": referral_rate,
        "human_review_rate": referral_rate,
    }


def evaluate_threshold_sweep(
    predictions_csv: str | Path,
    thresholds: Sequence[float] | None = None,
    positive_probability_column: str | None = None,
) -> dict[str, Any]:
    """Evaluate binary screening metrics across a threshold sweep."""

    normalized_thresholds = _validate_thresholds(thresholds)
    dataframe, resolved_probability_column = _load_predictions_csv(
        predictions_csv=predictions_csv,
        positive_probability_column=positive_probability_column,
    )

    if len(dataframe) == 0:
        raise ValueError("Predictions CSV must contain at least one row.")

    labels = torch.tensor(dataframe["true_label"].to_numpy(), dtype=torch.int64)
    positive_probabilities = torch.tensor(
        dataframe[resolved_probability_column].to_numpy(),
        dtype=torch.float32,
    )

    threshold_results = [
        _compute_threshold_metrics(labels, positive_probabilities, threshold)
        for threshold in normalized_thresholds
    ]

    return {
        "predictions_csv": str(predictions_csv),
        "positive_probability_column": resolved_probability_column,
        "threshold_results": threshold_results,
    }


def run_threshold_sweep_analysis(
    predictions_csv: str | Path,
    output_json: str | Path,
    output_csv: str | Path,
    thresholds: Sequence[float] | None = None,
    positive_probability_column: str | None = None,
) -> dict[str, Any]:
    """Run threshold-sweep analysis and write JSON/CSV outputs."""

    output_json_path = Path(output_json)
    output_csv_path = Path(output_csv)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    result = evaluate_threshold_sweep(
        predictions_csv=predictions_csv,
        thresholds=thresholds,
        positive_probability_column=positive_probability_column,
    )
    output_json_path.write_text(
        json.dumps(_serialize_for_json(result), indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(result["threshold_results"]).to_csv(output_csv_path, index=False)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for threshold-sweep analysis."""

    parser = argparse.ArgumentParser(
        description="Evaluate screening operating points across probability thresholds."
    )
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--positive-probability-column", type=str, default=None)
    parser.add_argument("--thresholds", type=float, nargs="*", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the threshold-sweep analysis CLI."""

    args = parse_args(argv)
    result = run_threshold_sweep_analysis(
        predictions_csv=args.predictions_csv,
        output_json=args.output_json,
        output_csv=args.output_csv,
        thresholds=args.thresholds,
        positive_probability_column=args.positive_probability_column,
    )
    print(f"Evaluated {len(result['threshold_results'])} thresholds.")
    print(f"Saved threshold sweep summary to: {args.output_json}")
    print(f"Saved threshold sweep table to: {args.output_csv}")


if __name__ == "__main__":
    main()
