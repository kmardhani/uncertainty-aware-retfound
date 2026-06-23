"""Validate that prepared APTOS metadata rows resolve to image files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.data.prepare_aptos_metadata import build_prepared_metadata
from uncertainty_retfound.data.aptos import (
    AptosImageValidationSummary,
    get_aptos_image_extension,
    get_aptos_image_root,
    validate_aptos_image_paths,
)
from uncertainty_retfound.utils.config import load_yaml_config


def validate_aptos_images(
    config_path: Path,
    metadata_csv: Path | None = None,
    task: str | None = None,
    max_missing_to_show: int = 20,
) -> AptosImageValidationSummary:
    """Validate APTOS image path existence from prepared or raw metadata."""

    dataset_config = load_yaml_config(config_path)
    metadata = (
        pd.read_csv(metadata_csv)
        if metadata_csv is not None
        else build_prepared_metadata(config_path, task=task)
    )

    image_root = get_aptos_image_root(dataset_config, config_path=config_path)
    image_extension = get_aptos_image_extension(dataset_config)
    preferred_id_column = dataset_config["labels"]["image_id_column"]

    return validate_aptos_image_paths(
        metadata,
        image_root=image_root,
        id_column=preferred_id_column,
        extension=image_extension,
        max_missing_to_show=max_missing_to_show,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Validate that APTOS metadata rows resolve to image files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/aptos2019.yaml"),
        help="Path to an APTOS-style dataset config YAML file.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional prepared metadata CSV to validate instead of preparing in memory.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional task name used when preparing metadata in memory.",
    )
    parser.add_argument(
        "--max-missing-to-show",
        type=int,
        default=20,
        help="Maximum number of missing image IDs and paths to print.",
    )
    return parser.parse_args()


def _print_summary(summary: AptosImageValidationSummary) -> None:
    """Print a concise validation summary."""

    print("APTOS image validation summary")
    print(f"Image root: {summary.image_root}")
    print(f"Image ID column: {summary.id_column}")
    print(f"Expected extension: {summary.extension}")
    print(f"Total rows: {summary.total_rows}")
    print(f"Found images: {summary.found_image_count}")
    print(f"Missing images: {summary.missing_image_count}")

    if summary.missing_image_count == 0:
        print("All image paths resolved successfully.")
        return

    print("Missing image IDs:")
    for image_id in summary.missing_image_ids:
        print(f"- {image_id}")

    print("Missing image paths:")
    for image_path in summary.missing_image_paths:
        print(f"- {image_path}")


def main() -> int:
    """Run APTOS image validation from the command line."""

    args = parse_args()
    summary = validate_aptos_images(
        config_path=args.config,
        metadata_csv=args.metadata_csv,
        task=args.task,
        max_missing_to_show=args.max_missing_to_show,
    )
    _print_summary(summary)
    return 0 if summary.all_images_found else 1


if __name__ == "__main__":
    raise SystemExit(main())
