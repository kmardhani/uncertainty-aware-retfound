"""Prepare APTOS metadata with task labels and train/val/test splits.

Example
-------
uv run python scripts/data/prepare_aptos_metadata.py \
  --config configs/datasets/fixtures/aptos2019_sample.yaml \
  --output data/processed/aptos2019_sample_metadata_splits.csv \
  --task referable_dr
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML config to contain a mapping: {path}")

    return config


def get_nested_value(config: dict[str, Any], key_path: tuple[str, ...]) -> Any | None:
    """Return a nested config value if it exists."""
    value: Any = config

    for key in key_path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]

    return value


def get_candidate_base_dirs(config: dict[str, Any], config_path: Path) -> list[Path]:
    """Return likely base directories for resolving relative dataset paths."""
    base_dirs = [Path.cwd(), config_path.parent]

    root_key_paths = [
        ("root_dir",),
        ("data_root",),
        ("dataset_root",),
        ("dataset", "root_dir"),
        ("dataset", "data_root"),
        ("paths", "root_dir"),
        ("paths", "data_root"),
        ("paths", "raw_dir"),
        ("data", "root_dir"),
        ("data", "data_root"),
        ("raw", "root_dir"),
    ]

    for key_path in root_key_paths:
        value = get_nested_value(config, key_path)

        if isinstance(value, str):
            root_path = Path(value)
            if not root_path.is_absolute():
                root_path = Path.cwd() / root_path
            base_dirs.append(root_path)

    unique_base_dirs: list[Path] = []
    for base_dir in base_dirs:
        if base_dir not in unique_base_dirs:
            unique_base_dirs.append(base_dir)

    return unique_base_dirs


def resolve_dataset_path(path_value: str | Path, config: dict[str, Any], config_path: Path) -> Path:
    """Resolve a dataset path against likely project/config/data roots."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    for base_dir in get_candidate_base_dirs(config, config_path):
        candidate = base_dir / path
        if candidate.exists():
            return candidate

    # Return the most likely repo-relative path even if it does not exist,
    # so downstream FileNotFoundError is still informative.
    return Path.cwd() / path


def find_csv_leaf_values(config: dict[str, Any]) -> list[str]:
    """Find CSV-like string values anywhere in the config."""
    csv_values: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str) and value.lower().endswith(".csv"):
            csv_values.append(value)

    walk(config)
    return csv_values


def get_metadata_csv_path(config: dict[str, Any], config_path: Path) -> Path:
    """Extract the APTOS metadata CSV path from the dataset config.

    Supports both explicit metadata keys and common dataset config layouts.
    """
    candidate_keys = [
        ("metadata_csv",),
        ("metadata_file",),
        ("metadata_path",),
        ("train_csv",),
        ("labels_csv",),
        ("csv_path",),
        ("metadata", "csv_path"),
        ("metadata", "path"),
        ("metadata", "file"),
        ("metadata", "filename"),
        ("dataset", "metadata_csv"),
        ("dataset", "metadata_file"),
        ("dataset", "train_csv"),
        ("paths", "metadata_csv"),
        ("paths", "metadata_file"),
        ("paths", "train_csv"),
        ("paths", "labels_csv"),
        ("raw", "train_csv"),
        ("raw", "metadata_csv"),
        ("raw", "metadata_file"),
        ("data", "train_csv"),
        ("data", "metadata_csv"),
        ("data", "metadata_file"),
    ]

    for key_path in candidate_keys:
        value = get_nested_value(config, key_path)

        if isinstance(value, str):
            return resolve_dataset_path(value, config, config_path)

    csv_values = find_csv_leaf_values(config)

    if len(csv_values) == 1:
        return resolve_dataset_path(csv_values[0], config, config_path)

    if len(csv_values) > 1:
        preferred_keywords = ("metadata", "train", "label", "aptos")

        for csv_value in csv_values:
            lower_value = csv_value.lower()
            if any(keyword in lower_value for keyword in preferred_keywords):
                return resolve_dataset_path(csv_value, config, config_path)

        raise KeyError(
            "Found multiple CSV paths in config but could not determine which one "
            f"is the metadata CSV: {csv_values}"
        )

    raise KeyError(
        "Could not find metadata CSV path in config. Expected a metadata/train/labels "
        "CSV path such as metadata_file, train_csv, paths.metadata_csv, "
        "dataset.metadata_file, or any CSV-valued config field."
    )


def get_selected_task(
    config: dict[str, Any],
    requested_task: str | None,
) -> tuple[str, dict[str, Any]]:
    """Return the selected task name and task config."""
    tasks = config.get("tasks") or config.get("task_mappings")

    if not isinstance(tasks, dict) or not tasks:
        raise KeyError("Config must define a non-empty 'tasks' or 'task_mappings' section.")

    task_name = requested_task or config.get("selected_task") or config.get("task")

    if task_name is None:
        if len(tasks) == 1:
            task_name = next(iter(tasks))
        else:
            available = ", ".join(tasks.keys())
            raise ValueError(
                "Multiple tasks are defined but no selected task was provided. "
                f"Use --task. Available tasks: {available}"
            )

    if task_name not in tasks:
        available = ", ".join(tasks.keys())
        raise KeyError(f"Task '{task_name}' not found. Available tasks: {available}")

    task_config = tasks[task_name]
    if not isinstance(task_config, dict):
        raise ValueError(f"Task config for '{task_name}' must be a mapping.")

    return str(task_name), task_config


def get_source_column(task_config: dict[str, Any]) -> str:
    """Return source label column from task config."""
    source_column = (
        task_config.get("source_column")
        or task_config.get("input_column")
        or task_config.get("from_column")
        or task_config.get("diagnosis_column")
        or "diagnosis"
    )

    return str(source_column)


def get_target_column(task_config: dict[str, Any]) -> str:
    """Return target label column from task config."""
    target_column = (
        task_config.get("target_column")
        or task_config.get("output_column")
        or task_config.get("to_column")
        or task_config.get("label_column")
        or "label"
    )

    return str(target_column)


def get_label_mapping(task_config: dict[str, Any]) -> dict[Any, Any] | None:
    """Return label mapping from task config if one exists."""
    mapping = (
        task_config.get("mapping")
        or task_config.get("label_mapping")
        or task_config.get("class_mapping")
        or task_config.get("diagnosis_mapping")
    )

    if mapping is None:
        return None

    if not isinstance(mapping, dict):
        raise ValueError("Task mapping must be a dictionary if provided.")

    normalized_mapping: dict[Any, Any] = {}

    for key, value in mapping.items():
        try:
            normalized_key: Any = int(key)
        except (TypeError, ValueError):
            normalized_key = key

        normalized_mapping[normalized_key] = value

    return normalized_mapping


def apply_task_mapping(
    df: pd.DataFrame,
    task_name: str,
    task_config: dict[str, Any],
) -> pd.DataFrame:
    """Apply task-specific label mapping to the metadata dataframe."""
    source_column = get_source_column(task_config)
    target_column = get_target_column(task_config)
    mapping = get_label_mapping(task_config)

    if source_column not in df.columns:
        raise KeyError(
            f"Source column '{source_column}' not found in metadata. "
            f"Available columns: {list(df.columns)}"
        )

    result = df.copy()

    if mapping is None:
        result[target_column] = result[source_column]
    else:
        result[target_column] = result[source_column].map(mapping)

        if result[target_column].isna().any():
            missing_values = sorted(result.loc[result[target_column].isna(), source_column].unique())
            raise ValueError(
                f"Task mapping for '{task_name}' does not cover source values: {missing_values}"
            )

    result["task"] = task_name
    return result


def get_split_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract split settings from config, with sensible defaults."""
    split_config = config.get("split") or config.get("splits") or {}

    if not isinstance(split_config, dict):
        raise ValueError("'split' or 'splits' config section must be a mapping.")

    return {
        "train_size": float(split_config.get("train_size", 0.70)),
        "val_size": float(split_config.get("val_size", 0.15)),
        "test_size": float(split_config.get("test_size", 0.15)),
        "seed": int(split_config.get("seed", 42)),
    }


def add_train_val_test_split(
    df: pd.DataFrame,
    label_column: str = "label",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Add deterministic train/val/test split labels to a metadata dataframe.

    Uses stratified splitting when possible. If the dataset is too small for
    stratification, it falls back to non-stratified splitting so fixture-based
    smoke tests can still run.
    """
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in dataframe.")

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Split sizes must sum to 1.0. Got train={train_size}, "
            f"val={val_size}, test={test_size}, total={total}"
        )

    if len(df) < 3:
        raise ValueError("At least 3 rows are required to create train/val/test splits.")

    result = df.copy()

    label_counts = result[label_column].value_counts()
    use_stratify = label_counts.min() >= 2

    train_df, temp_df = train_test_split(
        result,
        train_size=train_size,
        random_state=seed,
        stratify=result[label_column] if use_stratify else None,
    )

    relative_val_size = val_size / (val_size + test_size)

    temp_label_counts = temp_df[label_column].value_counts()
    use_temp_stratify = (
        len(temp_df) >= 2
        and temp_df[label_column].nunique() > 1
        and temp_label_counts.min() >= 2
    )

    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        random_state=seed,
        stratify=temp_df[label_column] if use_temp_stratify else None,
    )

    result["split"] = ""

    result.loc[train_df.index, "split"] = "train"
    result.loc[val_df.index, "split"] = "val"
    result.loc[test_df.index, "split"] = "test"

    if (result["split"] == "").any():
        raise RuntimeError("Some rows were not assigned to a split.")

    return result


def prepare_metadata(
    config_path: Path,
    output_path: Path,
    task: str | None = None,
) -> pd.DataFrame:
    """Prepare APTOS metadata and write the split CSV."""
    config = load_yaml(config_path)
    metadata_csv_path = get_metadata_csv_path(config, config_path)

    if not metadata_csv_path.exists():
        raise FileNotFoundError(f"APTOS metadata CSV not found: {metadata_csv_path}")

    df = pd.read_csv(metadata_csv_path)

    task_name, task_config = get_selected_task(config, task)
    df = apply_task_mapping(df, task_name, task_config)

    label_column = get_target_column(task_config)
    split_config = get_split_config(config)

    df = add_train_val_test_split(
        df,
        label_column=label_column,
        train_size=split_config["train_size"],
        val_size=split_config["val_size"],
        test_size=split_config["test_size"],
        seed=split_config["seed"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare APTOS metadata with task labels and train/val/test splits."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/aptos2019.yaml"),
        help="Path to dataset YAML config.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/aptos2019_metadata_splits.csv"),
        help="Output CSV path.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional task name. If omitted, uses selected_task/task from config.",
    )

    return parser.parse_args()


def main() -> None:
    """Run metadata preparation from the command line."""
    args = parse_args()

    df = prepare_metadata(
        config_path=args.config,
        output_path=args.output,
        task=args.task,
    )

    split_counts = df["split"].value_counts().to_dict()

    if "label" in df.columns:
        label_counts = df["label"].value_counts().sort_index().to_dict()
    else:
        label_columns = [column for column in df.columns if column.endswith("label")]
        label_counts = {
            column: df[column].value_counts().sort_index().to_dict()
            for column in label_columns
        }

    print(f"Saved prepared metadata to: {args.output}")
    print(f"Rows: {len(df)}")
    print(f"Split counts: {split_counts}")
    print(f"Label counts: {label_counts}")


if __name__ == "__main__":
    main()