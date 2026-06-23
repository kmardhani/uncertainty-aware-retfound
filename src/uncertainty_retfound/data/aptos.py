"""APTOS 2019 dataset metadata and image validation utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AptosImageValidationSummary:
    """Summary of APTOS image path existence checks."""

    total_rows: int
    found_image_count: int
    missing_image_count: int
    missing_image_ids: list[str]
    missing_image_paths: list[str]
    image_root: Path
    id_column: str
    extension: str

    @property
    def all_images_found(self) -> bool:
        """Return True when every metadata row resolves to an existing image."""
        return self.missing_image_count == 0


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


def _resolve_repo_relative_path(
    path_value: str | Path,
    config_path: str | Path | None = None,
) -> Path:
    """Resolve a configured path relative to the repo root or config file."""

    path = Path(path_value)

    if path.is_absolute():
        return path

    candidate_paths = [Path.cwd() / path]
    if config_path is not None:
        candidate_paths.append(Path(config_path).resolve().parent / path)

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    return candidate_paths[0]


def get_aptos_image_root(
    dataset_config: dict[str, Any],
    config_path: str | Path | None = None,
) -> Path:
    """Return the configured APTOS image directory."""

    try:
        image_dir = dataset_config["paths"]["image_dir"]
    except KeyError as error:
        raise KeyError("APTOS dataset config is missing paths.image_dir") from error

    return _resolve_repo_relative_path(image_dir, config_path=config_path)


def get_aptos_image_extension(
    dataset_config: dict[str, Any],
    default_extension: str = ".png",
) -> str:
    """Return the configured APTOS image filename extension."""

    candidate_extension = (
        dataset_config.get("paths", {}).get("image_extension")
        or dataset_config.get("images", {}).get("extension")
        or dataset_config.get("dataset", {}).get("image_extension")
        or default_extension
    )

    extension = str(candidate_extension).strip()
    if not extension:
        raise ValueError("APTOS image extension must not be empty.")

    if not extension.startswith("."):
        extension = f".{extension}"

    return extension


def infer_aptos_image_id_column(
    metadata: pd.DataFrame,
    preferred_column: str | None = None,
) -> str:
    """Infer the metadata column that stores the APTOS image identifier."""

    candidate_columns = [preferred_column, "image_id", "id_code"]

    for column in candidate_columns:
        if column and column in metadata.columns:
            return column

    raise KeyError(
        "Could not determine the APTOS image ID column. "
        f"Available columns: {list(metadata.columns)}"
    )


def resolve_aptos_image_paths(
    metadata: pd.DataFrame,
    image_root: str | Path,
    id_column: str | None = None,
    extension: str = ".png",
    output_column: str = "image_path",
) -> pd.DataFrame:
    """Resolve image file paths for APTOS metadata rows without loading pixels."""

    resolved_id_column = infer_aptos_image_id_column(metadata, preferred_column=id_column)
    normalized_extension = extension if extension.startswith(".") else f".{extension}"
    root_path = Path(image_root)

    def build_image_path(image_id: object) -> str:
        image_id_string = str(image_id)
        image_id_path = Path(image_id_string)
        filename = image_id_path.name

        if not Path(filename).suffix:
            filename = f"{filename}{normalized_extension}"

        return str(root_path / filename)

    output = metadata.copy()
    output[output_column] = output[resolved_id_column].map(build_image_path)
    return output


def validate_aptos_image_paths(
    metadata: pd.DataFrame,
    image_root: str | Path,
    id_column: str | None = None,
    extension: str = ".png",
    max_missing_to_show: int = 20,
) -> AptosImageValidationSummary:
    """Validate that APTOS image paths exist for every metadata row."""

    if max_missing_to_show < 0:
        raise ValueError("max_missing_to_show must be non-negative.")

    resolved_id_column = infer_aptos_image_id_column(metadata, preferred_column=id_column)
    resolved_metadata = resolve_aptos_image_paths(
        metadata,
        image_root=image_root,
        id_column=resolved_id_column,
        extension=extension,
    )

    image_paths = resolved_metadata["image_path"].map(Path)
    exists_mask = image_paths.map(Path.exists)
    missing_metadata = resolved_metadata.loc[~exists_mask]

    missing_image_ids = (
        missing_metadata[resolved_id_column].astype(str).head(max_missing_to_show).tolist()
    )
    missing_image_paths = (
        missing_metadata["image_path"].astype(str).head(max_missing_to_show).tolist()
    )

    total_rows = len(resolved_metadata)
    missing_image_count = int((~exists_mask).sum())

    return AptosImageValidationSummary(
        total_rows=total_rows,
        found_image_count=total_rows - missing_image_count,
        missing_image_count=missing_image_count,
        missing_image_ids=missing_image_ids,
        missing_image_paths=missing_image_paths,
        image_root=Path(image_root),
        id_column=resolved_id_column,
        extension=extension if extension.startswith(".") else f".{extension}",
    )
