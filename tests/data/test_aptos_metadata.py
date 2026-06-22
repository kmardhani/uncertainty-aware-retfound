"""Tests for APTOS metadata loading."""

from uncertainty_retfound.data.aptos import load_aptos_metadata
from uncertainty_retfound.utils.config import load_yaml_config


def test_load_aptos_metadata_referable_dr_fixture() -> None:
    dataset_config = load_yaml_config("configs/datasets/fixtures/aptos2019_sample.yaml")

    metadata = load_aptos_metadata(dataset_config, task_name="referable_dr")

    assert len(metadata) == 5
    assert metadata["image_id"].tolist() == [
        "sample_no_dr",
        "sample_mild",
        "sample_moderate",
        "sample_severe",
        "sample_proliferative",
    ]
    assert metadata["original_label"].tolist() == [0, 1, 2, 3, 4]
    assert metadata["mapped_label"].tolist() == [0, 0, 1, 1, 1]
    assert metadata["mapped_class_name"].tolist() == [
        "non_referable_dr",
        "non_referable_dr",
        "referable_dr",
        "referable_dr",
        "referable_dr",
    ]
    assert metadata["task"].unique().tolist() == ["referable_dr"]
    assert metadata["dataset"].unique().tolist() == ["aptos2019_sample"]


def test_load_aptos_metadata_binary_dr_fixture() -> None:
    dataset_config = load_yaml_config("configs/datasets/fixtures/aptos2019_sample.yaml")

    metadata = load_aptos_metadata(dataset_config, task_name="binary_dr")

    assert metadata["mapped_label"].tolist() == [0, 1, 1, 1, 1]
    assert metadata["mapped_class_name"].tolist() == [
        "no_dr",
        "any_dr",
        "any_dr",
        "any_dr",
        "any_dr",
    ]


def test_load_aptos_metadata_five_class_fixture() -> None:
    dataset_config = load_yaml_config("configs/datasets/fixtures/aptos2019_sample.yaml")

    metadata = load_aptos_metadata(dataset_config, task_name="five_class_dr")

    assert metadata["mapped_label"].tolist() == [0, 1, 2, 3, 4]
    assert metadata["mapped_class_name"].tolist() == [
        "no_dr",
        "mild_dr",
        "moderate_dr",
        "severe_dr",
        "proliferative_dr",
    ]