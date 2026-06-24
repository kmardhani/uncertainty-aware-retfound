"""Prepare DDR metadata with binary referable DR labels and train/val/test splits.

Example
-------
uv run python scripts/data/prepare_ddr_metadata.py \
  --labels-csv data/raw/ddr/DR_grading.csv \
  --image-root data/raw/ddr/DR_grading/DR_grading \
  --output-csv data/processed/ddr_referable_dr_metadata_splits.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from uncertainty_retfound.data.splits import add_train_val_test_split


_DEFAULT_LABELS_CSV = Path("data/raw/ddr/DR_grading.csv")
_DEFAULT_IMAGE_ROOT = Path("data/raw/ddr/DR_grading/DR_grading")
_DEFAULT_OUTPUT_CSV = Path("data/processed/ddr_referable_dr_metadata_splits.csv")
_ID_COLUMN_CANDIDATES: tuple[str, ...] = (
    "id_code",
    "image_id",
    "image",
    "image_name",
    "filename",
    "file_name",
    "img_id",
    "img",
)
_DIAGNOSIS_COLUMN_CANDIDATES: tuple[str, ...] = (
    "diagnosis",
    "dr_grade",
    "grade",
    "label",
    "dr_level",
)
_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)


def _resolve_column_name(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    description: str,
) -> str:
    """Resolve a required source column from a list of candidate names."""

    available_columns = set(dataframe.columns)
    for candidate in candidates:
        if candidate in available_columns:
            return candidate

    raise KeyError(
        f"Could not determine {description} column. "
        f"Tried {list(candidates)}. Available columns: {list(dataframe.columns)}"
    )


def _resolve_image_path(
    image_root: Path,
    raw_image_value: object,
) -> tuple[str, str] | None:
    """Resolve one DDR image path under the configured image root.

    Returns `(id_code, relative_image_path)` when found, otherwise `None`.
    """

    raw_text = str(raw_image_value).strip()
    if not raw_text:
        return None

    raw_path = Path(raw_text)
    candidate_relative_paths: list[Path] = []

    if raw_path.suffix:
        candidate_relative_paths.append(raw_path)
    else:
        candidate_relative_paths.extend(
            Path(f"{raw_text}{extension}") for extension in _IMAGE_EXTENSIONS
        )

    for relative_path in candidate_relative_paths:
        full_path = image_root / relative_path
        if full_path.exists():
            return relative_path.stem, relative_path.as_posix()

    return None


def _validate_split_sizes(val_size: float, test_size: float) -> float:
    """Validate val/test fractions and return the implied train fraction."""

    normalized_val_size = float(val_size)
    normalized_test_size = float(test_size)

    if normalized_val_size <= 0.0 or normalized_val_size >= 1.0:
        raise ValueError(f"val_size must be in the interval (0, 1). Got: {normalized_val_size}")
    if normalized_test_size <= 0.0 or normalized_test_size >= 1.0:
        raise ValueError(
            f"test_size must be in the interval (0, 1). Got: {normalized_test_size}"
        )

    train_size = 1.0 - normalized_val_size - normalized_test_size
    if train_size <= 0.0:
        raise ValueError(
            "val_size and test_size must leave a positive train fraction. "
            f"Got val_size={normalized_val_size}, test_size={normalized_test_size}."
        )

    return train_size


def build_prepared_metadata(
    labels_csv: str | Path = _DEFAULT_LABELS_CSV,
    image_root: str | Path = _DEFAULT_IMAGE_ROOT,
    *,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
    allow_missing_images: bool = False,
) -> pd.DataFrame:
    """Build standardized DDR metadata in memory."""

    labels_path = Path(labels_csv)
    image_root_path = Path(image_root)

    if not labels_path.exists():
        raise FileNotFoundError(f"DDR labels CSV not found: {labels_path}")
    if not image_root_path.exists():
        raise FileNotFoundError(f"DDR image root not found: {image_root_path}")

    train_size = _validate_split_sizes(val_size, test_size)

    raw_dataframe = pd.read_csv(labels_path)
    if raw_dataframe.empty:
        raise ValueError(f"DDR labels CSV is empty: {labels_path}")

    id_column = _resolve_column_name(
        raw_dataframe,
        _ID_COLUMN_CANDIDATES,
        description="image id",
    )
    diagnosis_column = _resolve_column_name(
        raw_dataframe,
        _DIAGNOSIS_COLUMN_CANDIDATES,
        description="diagnosis",
    )

    diagnosis_values = pd.to_numeric(raw_dataframe[diagnosis_column], errors="raise").astype(int)

    prepared_rows: list[dict[str, object]] = []
    missing_rows: list[tuple[str, str]] = []

    for row_index, raw_row in raw_dataframe.iterrows():
        raw_image_value = raw_row[id_column]
        resolved = _resolve_image_path(image_root_path, raw_image_value)
        raw_id_text = str(raw_image_value).strip()

        if resolved is None:
            missing_rows.append((raw_id_text, str(image_root_path)))
            continue

        diagnosis = int(diagnosis_values.loc[row_index])
        id_code, relative_image_path = resolved
        prepared_rows.append(
            {
                "id_code": id_code,
                "image_path": relative_image_path,
                "diagnosis": diagnosis,
                "label": 1 if diagnosis >= 2 else 0,
            }
        )

    if missing_rows and not allow_missing_images:
        preview = ", ".join(raw_id for raw_id, _ in missing_rows[:5])
        raise FileNotFoundError(
            f"Found {len(missing_rows)} labeled DDR images missing under {image_root_path}. "
            f"Examples: {preview}"
        )

    prepared_dataframe = pd.DataFrame(
        prepared_rows,
        columns=["id_code", "image_path", "diagnosis", "label"],
    )

    if prepared_dataframe.empty:
        raise ValueError("No labeled DDR rows remain after image-path validation.")

    split_dataframe = add_train_val_test_split(
        prepared_dataframe,
        split_config={
            "strategy": "stratified",
            "train_fraction": train_size,
            "val_fraction": float(val_size),
            "test_fraction": float(test_size),
            "random_seed": int(seed),
        },
        label_column="label",
        split_column="split",
    )

    return split_dataframe.loc[
        :,
        ["id_code", "image_path", "diagnosis", "label", "split"],
    ].reset_index(drop=True)


def write_prepared_metadata(dataframe: pd.DataFrame, output_csv: str | Path) -> None:
    """Write prepared DDR metadata to disk."""

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def prepare_metadata(
    labels_csv: str | Path = _DEFAULT_LABELS_CSV,
    image_root: str | Path = _DEFAULT_IMAGE_ROOT,
    output_csv: str | Path = _DEFAULT_OUTPUT_CSV,
    *,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
    allow_missing_images: bool = False,
) -> pd.DataFrame:
    """Prepare DDR metadata and write the output CSV."""

    dataframe = build_prepared_metadata(
        labels_csv=labels_csv,
        image_root=image_root,
        seed=seed,
        val_size=val_size,
        test_size=test_size,
        allow_missing_images=allow_missing_images,
    )
    write_prepared_metadata(dataframe, output_csv)
    return dataframe


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Prepare DDR metadata with binary referable DR labels and splits."
    )
    parser.add_argument("--labels-csv", type=Path, default=_DEFAULT_LABELS_CSV)
    parser.add_argument("--image-root", type=Path, default=_DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-csv", type=Path, default=_DEFAULT_OUTPUT_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--allow-missing-images", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run DDR metadata preparation from the command line."""

    args = parse_args(argv)
    dataframe = prepare_metadata(
        labels_csv=args.labels_csv,
        image_root=args.image_root,
        output_csv=args.output_csv,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
        allow_missing_images=bool(args.allow_missing_images),
    )

    print(f"Saved prepared metadata to: {args.output_csv}")
    print(f"Rows: {len(dataframe)}")
    print(f"Split counts: {dataframe['split'].value_counts().to_dict()}")
    print(f"Label counts: {dataframe['label'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
