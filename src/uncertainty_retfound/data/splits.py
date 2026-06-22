"""Utilities for creating reproducible dataset splits."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import sklearn.model_selection as sk_model_selection


def _train_test_split_dataframe(
    metadata: pd.DataFrame,
    train_size: float,
    random_seed: int,
    stratify: pd.Series[Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run sklearn train_test_split and return typed DataFrames."""

    split_result = cast(
        list[pd.DataFrame],
        sk_model_selection.train_test_split(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            metadata,
            train_size=train_size,
            random_state=random_seed,
            stratify=stratify,
        ),
    )

    return split_result[0], split_result[1]


def add_train_val_test_split(
    metadata: pd.DataFrame,
    split_config: dict[str, Any],
    label_column: str = "mapped_label",
    split_column: str = "split",
) -> pd.DataFrame:
    """Add a reproducible train/validation/test split column to metadata.

    Parameters
    ----------
    metadata:
        Metadata table containing at least the label column.
    split_config:
        Split configuration from the dataset YAML.
    label_column:
        Column to use for stratification.
    split_column:
        Name of the output split column.

    Returns
    -------
    pd.DataFrame
        Copy of metadata with a new split column.

    Raises
    ------
    ValueError
        If split fractions are invalid or stratified splitting is not possible.
    KeyError
        If the label column is missing.
    """

    if label_column not in metadata.columns:
        raise KeyError(f"Missing label column for splitting: {label_column}")

    strategy = split_config.get("strategy", "stratified")
    train_fraction = float(split_config["train_fraction"])
    val_fraction = float(split_config["val_fraction"])
    test_fraction = float(split_config["test_fraction"])
    random_seed = int(split_config.get("random_seed", 42))

    total_fraction = train_fraction + val_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-8:
        raise ValueError(
            "Split fractions must sum to 1.0. "
            f"Got train={train_fraction}, val={val_fraction}, "
            f"test={test_fraction}, total={total_fraction}."
        )

    if strategy != "stratified":
        raise ValueError(f"Unsupported split strategy: {strategy}")

    output = metadata.copy()

    labels = output[label_column]
    class_counts = labels.value_counts()

    if (class_counts < 2).any():
        raise ValueError(
            "Stratified splitting requires at least 2 samples per class. "
            f"Class counts: {class_counts.to_dict()}"
        )

    train_df, temp_df = _train_test_split_dataframe(
        metadata=output,
        train_size=train_fraction,
        random_seed=random_seed,
        stratify=labels,
    )

    relative_val_fraction = val_fraction / (val_fraction + test_fraction)

    temp_class_counts = temp_df[label_column].value_counts()
    use_second_stratify = not (temp_class_counts < 2).any()

    val_df, test_df = _train_test_split_dataframe(
        metadata=temp_df,
        train_size=relative_val_fraction,
        random_seed=random_seed,
        stratify=temp_df[label_column] if use_second_stratify else None,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[split_column] = "train"
    val_df[split_column] = "val"
    test_df[split_column] = "test"

    split_metadata = pd.concat([train_df, val_df, test_df], axis=0)
    split_metadata = split_metadata.sort_index()

    return split_metadata