"""Smoke test for APTOS metadata loading."""

#from pathlib import Path

from uncertainty_retfound.data.aptos import load_aptos_metadata
from uncertainty_retfound.utils.config import load_yaml_config


def main() -> None:
    dataset_config = load_yaml_config("configs/datasets/aptos2019.yaml")

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