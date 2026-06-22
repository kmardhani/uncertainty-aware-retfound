"""APTOS 2019 dataset metadata utilities."""

from pathlib import Path
from typing import Any

import pandas as pd


def load_aptos_metadata(
    dataset_config: dict[str, Any],
    task_name: str | None = None,
) -> pd.DataFrame:
    """Load APTOS metadata and apply the configured task label mapping.

    Parameters
    ----------
    dataset_config:
        Parsed dataset YAML configuration.
    task_name:
        Name of the task to use. If None, uses dataset_config["usage"]["initial_task"].

    Returns
    -------
    pd.DataFrame
        Metadata table with image ID, original label, mapped label, and class name.

    Raises
    ------
    FileNotFoundError
        If the configured labels CSV does not exist.
    KeyError
        If required config keys or CSV columns are missing.
    """

    task_name = task_name or dataset_config["usage"]["initial_task"]

    labels_csv = Path(dataset_config["paths"]["labels_csv"])
    image_id_column = dataset_config["labels"]["image_id_column"]
    original_label_column = dataset_config["labels"]["original_label_column"]

    if not labels_csv.exists():
        raise FileNotFoundError(
            f"APTOS labels CSV not found: {labels_csv}. "
            "Download the dataset or update configs/datasets/aptos2019.yaml."
        )

    metadata = pd.read_csv(labels_csv)

    required_columns = {image_id_column, original_label_column}
    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise KeyError(
            f"APTOS labels CSV is missing required columns: {sorted(missing_columns)}"
        )

    task_config = dataset_config["tasks"][task_name]
    label_mapping = {
        int(original_label): int(mapped_label)
        for original_label, mapped_label in task_config["mapping"].items()
    }
    class_names = {
        int(label): str(class_name)
        for label, class_name in task_config["class_names"].items()
    }

    output = metadata[[image_id_column, original_label_column]].copy()
    output = output.rename(
        columns={
            image_id_column: "image_id",
            original_label_column: "original_label",
        }
    )

    output["mapped_label"] = output["original_label"].map(label_mapping)
    output["mapped_class_name"] = output["mapped_label"].map(class_names)
    output["task"] = task_name
    output["dataset"] = dataset_config["dataset"]["name"]

    if output["mapped_label"].isna().any():
        missing_labels = sorted(
            output.loc[output["mapped_label"].isna(), "original_label"].unique()
        )
        raise ValueError(f"Unmapped original labels found: {missing_labels}")

    return output