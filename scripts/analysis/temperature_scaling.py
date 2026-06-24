"""Temperature scaling for saved baseline validation predictions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from uncertainty_retfound.evaluation.metrics import classification_summary


_REQUIRED_COLUMNS: tuple[str, ...] = (
    "true_label",
    "logit_class_0",
    "logit_class_1",
)


def _serialize_for_json(value: Any) -> Any:
    """Convert nested outputs into JSON-serializable values."""

    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_for_json(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_serialize_for_json(item) for item in value]
    return value


def _load_predictions_csv(predictions_csv: str | Path) -> pd.DataFrame:
    """Load a saved validation prediction CSV and validate required columns."""

    predictions_path = Path(predictions_csv)
    dataframe = pd.read_csv(predictions_path)
    missing_columns = [column for column in _REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"Predictions CSV is missing required logit columns: {missing}. "
            f"File: {predictions_path}"
        )
    return dataframe


def _extract_logits_and_labels(dataframe: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract binary logits and labels from a validation prediction dataframe."""

    logits = torch.tensor(
        dataframe.loc[:, ["logit_class_0", "logit_class_1"]].to_numpy(),
        dtype=torch.float32,
    )
    labels = torch.tensor(dataframe["true_label"].to_numpy(), dtype=torch.int64)
    return logits, labels


def _build_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    """Build reusable binary classification metrics from logits and labels."""

    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(logits, dim=1)
    return classification_summary(
        predictions=predictions,
        labels=labels,
        num_classes=2,
        positive_scores=probabilities[:, 1],
        probabilities=probabilities,
        logits=logits,
    )


def learn_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    initial_temperature: float = 1.0,
    max_iterations: int = 50,
) -> float:
    """Learn a positive temperature by minimizing NLL on saved logits."""

    if initial_temperature <= 0:
        raise ValueError(f"initial_temperature must be positive. Got: {initial_temperature}")

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Temperature scaling currently expects binary logits with shape [N, 2].")

    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("Labels must be a 1D tensor aligned with logits.")

    log_temperature = torch.nn.Parameter(torch.log(torch.tensor(initial_temperature)))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iterations,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).item())


def apply_temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits."""

    if temperature <= 0:
        raise ValueError(f"temperature must be positive. Got: {temperature}")
    return logits / temperature


def _build_metric_deltas(
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
) -> dict[str, float | None]:
    """Build metric deltas for comparable scalar metrics."""

    metric_deltas: dict[str, float | None] = {}
    candidate_keys = (
        "accuracy",
        "auc",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "balanced_accuracy",
        "brier_score",
        "negative_log_likelihood",
        "expected_calibration_error",
        "mean_confidence",
        "mean_positive_class_probability",
    )

    for key in candidate_keys:
        before_value = metrics_before.get(key)
        after_value = metrics_after.get(key)
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
            metric_deltas[key] = float(after_value) - float(before_value)
        else:
            metric_deltas[key] = None

    return metric_deltas


def build_temperature_scaled_predictions(
    dataframe: pd.DataFrame,
    temperature: float,
) -> pd.DataFrame:
    """Return a copy of predictions with temperature-scaled probability columns appended."""

    logits, labels = _extract_logits_and_labels(dataframe)
    scaled_logits = apply_temperature_scaling(logits, temperature)
    scaled_probabilities = torch.softmax(scaled_logits, dim=1)
    scaled_predictions = torch.argmax(scaled_logits, dim=1)
    scaled_confidence = torch.max(scaled_probabilities, dim=1).values

    scaled_dataframe = dataframe.copy()
    scaled_dataframe["temperature_scaled_probability_class_0"] = scaled_probabilities[:, 0].tolist()
    scaled_dataframe["temperature_scaled_probability_class_1"] = scaled_probabilities[:, 1].tolist()
    scaled_dataframe["temperature_scaled_confidence"] = scaled_confidence.tolist()
    scaled_dataframe["temperature_scaled_predicted_label"] = scaled_predictions.tolist()
    scaled_dataframe["temperature_scaled_is_correct"] = (scaled_predictions == labels).tolist()
    return scaled_dataframe


def run_temperature_scaling(
    predictions_csv: str | Path,
    output_json: str | Path,
    output_predictions_csv: str | Path,
) -> dict[str, Any]:
    """Fit temperature scaling on saved validation predictions and write outputs."""

    predictions_path = Path(predictions_csv)
    dataframe = _load_predictions_csv(predictions_path)
    logits, labels = _extract_logits_and_labels(dataframe)

    metrics_before = _build_metrics_from_logits(logits, labels)
    temperature = learn_temperature(logits, labels)
    scaled_logits = apply_temperature_scaling(logits, temperature)
    metrics_after = _build_metrics_from_logits(scaled_logits, labels)
    metric_deltas = _build_metric_deltas(metrics_before, metrics_after)

    result = {
        "learned_temperature": float(temperature),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "metric_deltas": metric_deltas,
        "num_examples": int(labels.shape[0]),
        "source_predictions_csv": str(predictions_path),
    }

    output_json_path = Path(output_json)
    output_json_path.write_text(
        json.dumps(_serialize_for_json(result), indent=2),
        encoding="utf-8",
    )

    scaled_predictions = build_temperature_scaled_predictions(dataframe, temperature)
    output_predictions_path = Path(output_predictions_csv)
    scaled_predictions.to_csv(output_predictions_path, index=False)

    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for temperature scaling."""

    parser = argparse.ArgumentParser(description="Apply temperature scaling to validation logits.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions-csv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the temperature scaling CLI."""

    args = parse_args(argv)
    result = run_temperature_scaling(
        predictions_csv=args.predictions_csv,
        output_json=args.output_json,
        output_predictions_csv=args.output_predictions_csv,
    )
    print(f"Learned temperature: {result['learned_temperature']:.6f}")
    print(f"Saved calibration summary to: {args.output_json}")
    print(f"Saved scaled predictions to: {args.output_predictions_csv}")


if __name__ == "__main__":
    main()
