"""Smoke test for APTOS metadata loading."""

import argparse

from uncertainty_retfound.data.aptos import load_aptos_metadata
from uncertainty_retfound.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check APTOS metadata loading.")
    parser.add_argument(
        "--config",
        default="configs/datasets/aptos2019.yaml",
        help="Path to an APTOS-style dataset config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = load_yaml_config(args.config)

    try:
        metadata = load_aptos_metadata(dataset_config)
    except FileNotFoundError as error:
        print("APTOS metadata smoke test skipped")
        print(error)
        return

    print("APTOS metadata loaded successfully")
    print(f"Rows: {len(metadata)}")
    print("Columns:")
    print(metadata.columns.tolist())
    print("Mapped label distribution:")
    print(metadata["mapped_class_name"].value_counts())


if __name__ == "__main__":
    main()