"""Tests for dataset split utilities."""

import pandas as pd
import pytest

from typing import Any

from uncertainty_retfound.data.splits import add_train_val_test_split


def test_add_train_val_test_split_creates_expected_split_column() -> None:
    metadata = pd.DataFrame(
        {
            "image_id": [f"image_{i}" for i in range(20)],
            "mapped_label": [0] * 10 + [1] * 10,
        }
    )

    split_config: dict[str, Any] = {
        "strategy": "stratified",
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "random_seed": 42,
    }

    result = add_train_val_test_split(metadata, split_config)

    assert "split" in result.columns
    assert len(result) == 20
    assert set(result["split"].unique()) == {"train", "val", "test"}

    split_counts = result["split"].value_counts().to_dict()

    assert split_counts["train"] == 14
    assert split_counts["val"] == 3
    assert split_counts["test"] == 3


def test_add_train_val_test_split_preserves_all_rows() -> None:
    metadata = pd.DataFrame(
        {
            "image_id": [f"image_{i}" for i in range(20)],
            "mapped_label": [0] * 10 + [1] * 10,
        }
    )

    split_config: dict[str, Any] = {
        "strategy": "stratified",
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "random_seed": 42,
    }

    result = add_train_val_test_split(metadata, split_config)

    assert sorted(result["image_id"].tolist()) == sorted(metadata["image_id"].tolist())


def test_add_train_val_test_split_is_reproducible() -> None:
    metadata = pd.DataFrame(
        {
            "image_id": [f"image_{i}" for i in range(20)],
            "mapped_label": [0] * 10 + [1] * 10,
        }
    )

    split_config: dict[str, Any] = {
        "strategy": "stratified",
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "random_seed": 42,
    }

    result_1 = add_train_val_test_split(metadata, split_config)
    result_2 = add_train_val_test_split(metadata, split_config)

    assert result_1["split"].tolist() == result_2["split"].tolist()


def test_add_train_val_test_split_rejects_invalid_fractions() -> None:
    metadata = pd.DataFrame(
        {
            "image_id": [f"image_{i}" for i in range(20)],
            "mapped_label": [0] * 10 + [1] * 10,
        }
    )

    split_config: dict[str, Any] = {
        "strategy": "stratified",
        "train_fraction": 0.80,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "random_seed": 42,
    }

    with pytest.raises(ValueError, match="Split fractions must sum to 1.0"):
        add_train_val_test_split(metadata, split_config)


def test_add_train_val_test_split_rejects_missing_label_column() -> None:
    metadata = pd.DataFrame(
        {
            "image_id": [f"image_{i}" for i in range(20)],
        }
    )

    split_config: dict[str, Any] = {
        "strategy": "stratified",
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "random_seed": 42,
    }

    with pytest.raises(KeyError, match="Missing label column"):
        add_train_val_test_split(metadata, split_config)


def test_add_train_val_test_split_rejects_too_few_samples_per_class() -> None:
    metadata = pd.DataFrame(
        {
            "image_id": ["image_0", "image_1", "image_2"],
            "mapped_label": [0, 1, 1],
        }
    )

    split_config: dict[str, Any] = {
        "strategy": "stratified",
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "random_seed": 42,
    }

    with pytest.raises(ValueError, match="at least 2 samples per class"):
        add_train_val_test_split(metadata, split_config)