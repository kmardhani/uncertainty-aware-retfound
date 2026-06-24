"""Evaluate selective referral / defer-to-human policies from validation predictions."""

from __future__ import annotations

import argparse
import json
import math
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


_REQUIRED_COLUMNS: tuple[str, ...] = ("true_label", "predicted_label")


def _serialize_for_json(value: Any) -> Any:
    """Convert nested outputs into JSON-serializable values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_for_json(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_serialize_for_json(item) for item in value]
    return value


def _load_predictions_csv(
    predictions_csv: str | Path,
    uncertainty_column: str,
) -> pd.DataFrame:
    """Load a saved validation prediction CSV and validate required columns."""

    predictions_path = Path(predictions_csv)
    dataframe = pd.read_csv(predictions_path)
    missing_columns = [
        column
        for column in (*_REQUIRED_COLUMNS, uncertainty_column)
        if column not in dataframe.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"Predictions CSV is missing required columns: {missing}. File: {predictions_path}"
        )
    return dataframe.copy()


def _validate_coverage_levels(coverage_levels: Sequence[float]) -> list[float]:
    """Validate and normalize coverage levels."""

    if not coverage_levels:
        raise ValueError("At least one coverage level is required.")

    normalized_levels: list[float] = []
    for coverage in coverage_levels:
        normalized_coverage = float(coverage)
        if normalized_coverage <= 0.0 or normalized_coverage > 1.0:
            raise ValueError(
                f"Coverage levels must be in the interval (0, 1]. Got: {normalized_coverage}"
            )
        normalized_levels.append(normalized_coverage)

    return normalized_levels


def _compute_selective_metrics(
    accepted_dataframe: pd.DataFrame,
    total_examples: int,
    coverage_level: float,
) -> dict[str, float | int]:
    """Compute binary metrics over accepted examples."""

    accepted_count = len(accepted_dataframe)
    deferred_count = total_examples - accepted_count

    if accepted_count == 0:
        raise ValueError(
            f"Coverage level {coverage_level} yields zero accepted examples; "
            "select a higher coverage level or larger validation set."
        )

    labels = torch.tensor(accepted_dataframe["true_label"].to_numpy(), dtype=torch.int64)
    predictions = torch.tensor(
        accepted_dataframe["predicted_label"].to_numpy(),
        dtype=torch.int64,
    )

    true_positive = int(((predictions == 1) & (labels == 1)).sum().item())
    true_negative = int(((predictions == 0) & (labels == 0)).sum().item())
    false_positive = int(((predictions == 1) & (labels == 0)).sum().item())
    false_negative = int(((predictions == 0) & (labels == 1)).sum().item())
    referral_rate = float((predictions == 1).float().mean().item())

    return {
        "coverage": float(accepted_count / total_examples),
        "target_coverage": float(coverage_level),
        "deferred_count": deferred_count,
        "accepted_count": accepted_count,
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
    }


def evaluate_selective_referral(
    predictions_csv: str | Path,
    uncertainty_column: str,
    higher_is_more_uncertain: bool,
    coverage_levels: Sequence[float],
) -> list[dict[str, float | int]]:
    """Evaluate defer-to-human policies across multiple coverage levels."""

    normalized_coverages = _validate_coverage_levels(coverage_levels)
    dataframe = _load_predictions_csv(predictions_csv, uncertainty_column)
    total_examples = len(dataframe)

    if total_examples == 0:
        raise ValueError("Predictions CSV must contain at least one row.")

    ascending = higher_is_more_uncertain
    sorted_dataframe = dataframe.sort_values(
        by=[uncertainty_column, "id_code"] if "id_code" in dataframe.columns else [uncertainty_column],
        ascending=[ascending, True] if "id_code" in dataframe.columns else [ascending],
        kind="mergesort",
    ).reset_index(drop=True)

    results: list[dict[str, float | int]] = []
    for coverage_level in normalized_coverages:
        accepted_count = int(math.ceil(total_examples * coverage_level))
        accepted_count = min(max(accepted_count, 1), total_examples)
        accepted_dataframe = sorted_dataframe.iloc[:accepted_count].copy()
        results.append(
            _compute_selective_metrics(
                accepted_dataframe=accepted_dataframe,
                total_examples=total_examples,
                coverage_level=coverage_level,
            )
        )

    return results


def run_selective_referral_analysis(
    predictions_csv: str | Path,
    uncertainty_column: str,
    higher_is_more_uncertain: bool,
    coverage_levels: Sequence[float],
    output_json: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    """Run selective referral analysis and write JSON/CSV outputs."""

    output_json_path = Path(output_json)
    output_csv_path = Path(output_csv)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    coverage_results = evaluate_selective_referral(
        predictions_csv=predictions_csv,
        uncertainty_column=uncertainty_column,
        higher_is_more_uncertain=higher_is_more_uncertain,
        coverage_levels=coverage_levels,
    )

    result = {
        "predictions_csv": str(predictions_csv),
        "uncertainty_column": uncertainty_column,
        "higher_is_more_uncertain": bool(higher_is_more_uncertain),
        "coverage_results": coverage_results,
    }
    output_json_path.write_text(
        json.dumps(_serialize_for_json(result), indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(coverage_results).to_csv(output_csv_path, index=False)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for selective referral analysis."""

    parser = argparse.ArgumentParser(
        description="Evaluate selective referral / defer-to-human policies."
    )
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--uncertainty-column", type=str, required=True)
    direction_group = parser.add_mutually_exclusive_group(required=True)
    direction_group.add_argument("--higher-is-more-uncertain", action="store_true")
    direction_group.add_argument("--lower-is-more-uncertain", action="store_true")
    parser.add_argument(
        "--coverage-levels",
        type=float,
        nargs="+",
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selective referral analysis CLI."""

    args = parse_args(argv)
    result = run_selective_referral_analysis(
        predictions_csv=args.predictions_csv,
        uncertainty_column=args.uncertainty_column,
        higher_is_more_uncertain=bool(args.higher_is_more_uncertain),
        coverage_levels=args.coverage_levels,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(f"Evaluated {len(result['coverage_results'])} coverage levels.")
    print(f"Saved selective referral summary to: {args.output_json}")
    print(f"Saved selective referral table to: {args.output_csv}")


if __name__ == "__main__":
    main()
