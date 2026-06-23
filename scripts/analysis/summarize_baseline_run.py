"""Summarize baseline training outputs without modifying any files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


_EPOCH_COLUMNS: tuple[tuple[str, str], ...] = (
    ("epoch", "epoch"),
    ("train_loss", "train_loss"),
    ("val_loss", "val_loss"),
    ("accuracy", "val_accuracy"),
    ("auc", "auc"),
    ("precision", "precision"),
    ("recall/sensitivity", "sensitivity"),
    ("specificity", "specificity"),
    ("f1", "f1"),
    ("balanced_accuracy", "balanced_accuracy"),
    ("brier_score", "brier_score"),
    ("negative_log_likelihood", "negative_log_likelihood"),
    ("expected_calibration_error", "expected_calibration_error"),
    ("mean_confidence", "mean_confidence"),
    ("mean_positive_class_probability", "mean_positive_class_probability"),
)


def _require_metrics_path(output_dir: Path) -> Path:
    """Return the metrics path or raise a clear error if it is missing."""

    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in output directory: {output_dir}")
    return metrics_path


def _load_metrics(output_dir: Path) -> dict[str, Any]:
    """Load metrics.json from an output directory."""

    metrics_path = _require_metrics_path(output_dir)
    raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(raw_metrics, dict):
        raise ValueError(f"metrics.json must contain a JSON object: {metrics_path}")
    return raw_metrics


def _load_validation_predictions(output_dir: Path) -> pd.DataFrame | None:
    """Load validation_predictions.csv when it exists."""

    predictions_path = output_dir / "validation_predictions.csv"
    if not predictions_path.exists():
        return None
    return pd.read_csv(predictions_path)


def _format_value(value: Any) -> str:
    """Format a scalar value for text output."""

    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_json_block(value: Any) -> str:
    """Format nested JSON-like values for readable output."""

    return json.dumps(value, indent=2, sort_keys=True)


def _extract_epoch_metric(epoch_result: Mapping[str, Any], metric_key: str) -> Any:
    """Extract an epoch-level metric from the saved result structure."""

    if metric_key == "epoch":
        return epoch_result.get("epoch")

    if metric_key == "val_accuracy":
        return epoch_result.get("val_accuracy")

    if metric_key in ("train_loss", "val_loss"):
        return epoch_result.get(metric_key)

    val_metrics = epoch_result.get("val_metrics")
    if isinstance(val_metrics, Mapping):
        return val_metrics.get(metric_key)

    return None


def _build_epoch_table(epoch_results: Sequence[Mapping[str, Any]]) -> str:
    """Build a plain-text table over epoch results."""

    rows: list[list[str]] = []
    header = [column_name for column_name, _ in _EPOCH_COLUMNS]

    for epoch_result in epoch_results:
        row = [
            _format_value(_extract_epoch_metric(epoch_result, metric_key))
            for _, metric_key in _EPOCH_COLUMNS
        ]
        rows.append(row)

    widths = [len(column) for column in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    header_line = " | ".join(
        column.ljust(widths[index]) for index, column in enumerate(header)
    )
    separator_line = "-+-".join("-" * width for width in widths)
    row_lines = [
        " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) for row in rows
    ]

    return "\n".join([header_line, separator_line, *row_lines])


def _best_epoch_line(
    epoch_results: Sequence[Mapping[str, Any]],
    label: str,
    metric_key: str,
    reverse: bool,
) -> str | None:
    """Return a one-line best-epoch summary for a metric."""

    candidates: list[tuple[float, Mapping[str, Any]]] = []
    for epoch_result in epoch_results:
        metric_value = _extract_epoch_metric(epoch_result, metric_key)
        if isinstance(metric_value, (int, float)):
            candidates.append((float(metric_value), epoch_result))

    if not candidates:
        return None

    best_value, best_epoch = sorted(candidates, key=lambda item: item[0], reverse=reverse)[0]
    epoch_number = best_epoch.get("epoch", "n/a")
    return f"- {label}: epoch {epoch_number} ({metric_key}={best_value:.4f})"


def summarize_baseline_run(output_dir: str | Path) -> str:
    """Build a read-only text summary for a baseline training run."""

    output_dir_path = Path(output_dir)
    metrics = _load_metrics(output_dir_path)
    predictions = _load_validation_predictions(output_dir_path)

    epoch_results_object = metrics.get("epoch_results", [])
    if not isinstance(epoch_results_object, list):
        raise ValueError("metrics.json field 'epoch_results' must be a list.")

    epoch_results = [
        epoch_result
        for epoch_result in epoch_results_object
        if isinstance(epoch_result, Mapping)
    ]

    lines = [
        "Run Configuration",
        f"- model_type: {_format_value(metrics.get('model_type'))}",
        f"- output_dir: {_format_value(metrics.get('output_dir', output_dir_path))}",
        f"- epochs: {_format_value(metrics.get('epochs'))}",
        f"- batch_size: {_format_value(metrics.get('batch_size'))}",
        f"- learning_rate: {_format_value(metrics.get('learning_rate'))}",
        f"- device: {_format_value(metrics.get('device'))}",
    ]

    if metrics.get("backbone_checkpoint") is not None:
        lines.append(f"- backbone_checkpoint: {_format_value(metrics.get('backbone_checkpoint'))}")
    if metrics.get("retfound_repo_path") is not None:
        lines.append(f"- retfound_repo_path: {_format_value(metrics.get('retfound_repo_path'))}")

    lines.append("")
    lines.append("Epoch Results")
    if epoch_results:
        lines.append(_build_epoch_table(epoch_results))
    else:
        lines.append("No epoch results found.")

    lines.append("")
    lines.append("Best Epochs")
    best_epoch_lines = [
        _best_epoch_line(epoch_results, "Highest accuracy", "val_accuracy", reverse=True),
        _best_epoch_line(epoch_results, "Lowest validation loss", "val_loss", reverse=False),
        _best_epoch_line(epoch_results, "Highest recall/sensitivity", "sensitivity", reverse=True),
        _best_epoch_line(
            epoch_results,
            "Highest balanced_accuracy",
            "balanced_accuracy",
            reverse=True,
        ),
        _best_epoch_line(
            epoch_results,
            "Lowest expected_calibration_error",
            "expected_calibration_error",
            reverse=False,
        ),
    ]
    lines.extend(line for line in best_epoch_lines if line is not None)
    if not any(line is not None for line in best_epoch_lines):
        lines.append("No comparable epoch metrics found.")

    lines.append("")
    lines.append("Final Validation Metrics")
    lines.append(_format_json_block(metrics.get("final_validation_metrics", {})))

    lines.append("")
    lines.append("Validation Predictions")
    if predictions is None:
        lines.append("validation_predictions.csv not found.")
    else:
        lines.append(f"- row_count: {len(predictions)}")
        lines.append(f"- columns: {', '.join(predictions.columns.tolist())}")
        if "is_correct" in predictions.columns:
            correct_count = int(predictions["is_correct"].astype(bool).sum())
            incorrect_count = int(len(predictions) - correct_count)
            lines.append(f"- correct: {correct_count}")
            lines.append(f"- incorrect: {incorrect_count}")
        if "confidence" in predictions.columns:
            mean_confidence = float(predictions["confidence"].mean())
            lines.append(f"- average_confidence: {mean_confidence:.4f}")

    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the summary script."""

    parser = argparse.ArgumentParser(description="Summarize a baseline training output directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the baseline training output summary CLI."""

    args = parse_args(argv)
    print(summarize_baseline_run(args.output_dir))


if __name__ == "__main__":
    main()
