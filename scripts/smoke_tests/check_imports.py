"""Basic smoke test for imports and YAML configuration loading."""

from pathlib import Path

import uncertainty_retfound
from uncertainty_retfound.utils.config import load_yaml_config


def main() -> None:
    print("uncertainty_retfound import successful")
    print(f"Package: {uncertainty_retfound.__name__}")

    experiment_config_path = Path("configs/experiments/smoke_test.yaml")
    experiment_config = load_yaml_config(experiment_config_path)

    dataset_config_path = Path(experiment_config["dataset"]["config_path"])
    dataset_config = load_yaml_config(dataset_config_path)

    print("Experiment config loaded successfully")
    print(f"Experiment: {experiment_config['experiment']['name']}")

    print("Dataset config loaded successfully")
    print(f"Dataset: {dataset_config['dataset']['name']}")
    print(f"Initial task: {dataset_config['usage']['initial_task']}")


if __name__ == "__main__":
    main()