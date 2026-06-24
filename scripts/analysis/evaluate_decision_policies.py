"""Evaluate clinical decision policies from saved validation prediction CSV files."""

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


_REQUIRED_COLUMNS: tuple[str, ...] = ("id_code", "true_label", "predicted_label")


def _serialize_for_json(value: Any) -> Any:
    """Convert nested outputs into JSON-serializable values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_for_json(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_serialize_for_json(item) for item in value]
    return value


def _parse_name_path_pair(raw_value: str) -> tuple[str, Path]:
    """Parse a `name=path` CLI argument."""

    if "=" not in raw_value:
        raise ValueError(
            "Each --predictions argument must use the form name=path/to/file.csv. "
            f"Got: {raw_value}"
        )

    name, path_value = raw_value.split("=", 1)
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError(f"Prediction source name must not be empty: {raw_value}")

    predictions_path = Path(path_value).expanduser()
    return normalized_name, predictions_path


def _load_prediction_csv(name: str, predictions_path: Path) -> pd.DataFrame:
    """Load and validate one saved validation prediction CSV."""

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found for '{name}': {predictions_path}")

    dataframe = pd.read_csv(predictions_path)
    missing_columns = [column for column in _REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"Predictions CSV for '{name}' is missing required columns: {missing}. "
            f"File: {predictions_path}"
        )

    return dataframe.copy()


def merge_prediction_csvs(prediction_sources: dict[str, Path]) -> pd.DataFrame:
    """Merge multiple prediction CSVs on `id_code` with label consistency checks."""

    if not prediction_sources:
        raise ValueError("At least one prediction source is required.")

    merged_dataframe: pd.DataFrame | None = None

    for index, (name, predictions_path) in enumerate(prediction_sources.items()):
        dataframe = _load_prediction_csv(name, predictions_path)
        rename_mapping = {
            "predicted_label": f"predicted_label__{name}",
        }
        if "image_path" in dataframe.columns:
            rename_mapping["image_path"] = f"image_path__{name}"

        selected_columns = ["id_code", "true_label", "predicted_label"]
        if "image_path" in dataframe.columns:
            selected_columns.append("image_path")

        renamed = dataframe.loc[:, selected_columns].rename(columns=rename_mapping)

        if index == 0:
            merged_dataframe = renamed
            continue

        assert merged_dataframe is not None
        previous_row_count = len(merged_dataframe)
        current_row_count = len(renamed)
        merged_dataframe = merged_dataframe.merge(
            renamed,
            on="id_code",
            how="inner",
            suffixes=("", f"__{name}"),
        )

        if len(merged_dataframe) != previous_row_count or len(merged_dataframe) != current_row_count:
            raise ValueError("Prediction CSV files do not contain matching id_code sets.")

        previous_true_label = merged_dataframe["true_label"]
        current_true_label_column = f"true_label__{name}"
        if current_true_label_column not in merged_dataframe.columns:
            raise ValueError(f"Failed to track true_label column for prediction source '{name}'.")

        current_true_label = merged_dataframe.pop(current_true_label_column)
        if not previous_true_label.equals(current_true_label):
            raise ValueError("Prediction CSV files have mismatched true_label values across id_code.")

    if merged_dataframe is None:
        raise RuntimeError("Failed to merge prediction CSV files.")

    return merged_dataframe.sort_values("id_code").reset_index(drop=True)


def _compute_policy_metrics(
    policy_name: str,
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float | int | str]:
    """Compute binary classification and operating metrics for one policy."""

    true_positive = int(((predictions == 1) & (labels == 1)).sum().item())
    true_negative = int(((predictions == 0) & (labels == 0)).sum().item())
    false_positive = int(((predictions == 1) & (labels == 0)).sum().item())
    false_negative = int(((predictions == 0) & (labels == 1)).sum().item())
    referral_rate = float((predictions == 1).float().mean().item())

    return {
        "policy": policy_name,
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


def evaluate_decision_policies(
    prediction_sources: dict[str, Path],
) -> list[dict[str, float | int | str]]:
    """Evaluate individual and ensemble decision policies from saved prediction CSV files."""

    merged = merge_prediction_csvs(prediction_sources)
    labels = torch.tensor(merged["true_label"].to_numpy(), dtype=torch.int64)

    prediction_columns = [
        f"predicted_label__{name}" for name in prediction_sources
    ]
    prediction_matrix = torch.tensor(
        merged.loc[:, prediction_columns].to_numpy(),
        dtype=torch.int64,
    )

    policy_results: list[dict[str, float | int | str]] = []

    for column_name, source_name in zip(prediction_columns, prediction_sources, strict=True):
        source_predictions = torch.tensor(merged[column_name].to_numpy(), dtype=torch.int64)
        policy_results.append(
            _compute_policy_metrics(
                policy_name=f"model:{source_name}",
                predictions=source_predictions,
                labels=labels,
            )
        )

    any_positive_predictions = (prediction_matrix == 1).any(dim=1).to(dtype=torch.int64)
    all_positive_predictions = (prediction_matrix == 1).all(dim=1).to(dtype=torch.int64)
    majority_predictions = (
        prediction_matrix.sum(dim=1).to(dtype=torch.float32)
        >= (prediction_matrix.shape[1] / 2.0)
    ).to(dtype=torch.int64)

    policy_results.append(
        _compute_policy_metrics("policy:or_rule", any_positive_predictions, labels)
    )
    policy_results.append(
        _compute_policy_metrics("policy:majority_vote", majority_predictions, labels)
    )
    policy_results.append(
        _compute_policy_metrics("policy:and_rule", all_positive_predictions, labels)
    )

    return policy_results


def run_policy_analysis(
    prediction_pairs: Sequence[str],
    output_json: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    """Run the decision-policy analysis and write JSON/CSV outputs."""

    if not prediction_pairs:
        raise ValueError("At least one --predictions name=path pair is required.")

    prediction_sources = dict(_parse_name_path_pair(raw_value) for raw_value in prediction_pairs)
    policy_results = evaluate_decision_policies(prediction_sources)

    output_json_path = Path(output_json)
    output_csv_path = Path(output_csv)

    result = {
        "prediction_sources": {name: str(path) for name, path in prediction_sources.items()},
        "policy_results": policy_results,
    }

    output_json_path.write_text(
        json.dumps(_serialize_for_json(result), indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(policy_results).to_csv(output_csv_path, index=False)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for decision-policy analysis."""

    parser = argparse.ArgumentParser(
        description="Evaluate single-model and ensemble decision policies from validation predictions."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        action="append",
        required=True,
        help="Repeatable name=path pair for a validation_predictions.csv file.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the decision-policy analysis CLI."""

    args = parse_args(argv)
    result = run_policy_analysis(
        prediction_pairs=args.predictions,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(f"Evaluated {len(result['policy_results'])} policies.")
    print(f"Saved policy summary to: {args.output_json}")
    print(f"Saved policy table to: {args.output_csv}")


if __name__ == "__main__":
    main()
