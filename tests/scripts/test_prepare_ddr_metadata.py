"""Tests for the DDR metadata preparation script."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.data.prepare_ddr_metadata import prepare_metadata


def _write_labels_csv(labels_csv: Path, image_names: list[str], diagnoses: list[int]) -> None:
    dataframe = pd.DataFrame(
        {
            "image_id": image_names,
            "diagnosis": diagnoses,
        }
    )
    dataframe.to_csv(labels_csv, index=False)


def _touch_images(image_root: Path, image_names: list[str]) -> None:
    image_root.mkdir(parents=True, exist_ok=True)
    for image_name in image_names:
        (image_root / image_name).write_bytes(b"fixture")


def test_prepare_metadata_writes_expected_ddr_output(tmp_path: Path) -> None:
    labels_csv = tmp_path / "DR_grading.csv"
    image_root = tmp_path / "DR_grading"
    output_csv = tmp_path / "ddr_metadata.csv"

    image_names = [f"image_{index:02d}.jpg" for index in range(12)]
    diagnoses = [0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4, 2]

    _write_labels_csv(labels_csv, image_names, diagnoses)
    _touch_images(image_root, image_names + ["extra_a.jpg", "extra_b.jpg"])

    dataframe = prepare_metadata(
        labels_csv=labels_csv,
        image_root=image_root,
        output_csv=output_csv,
        seed=7,
    )

    written_dataframe = pd.read_csv(output_csv)

    assert output_csv.exists()
    assert len(dataframe) == 12
    assert len(written_dataframe) == 12
    assert list(dataframe.columns) == ["id_code", "image_path", "diagnosis", "label", "split"]
    assert list(written_dataframe.columns) == [
        "id_code",
        "image_path",
        "diagnosis",
        "label",
        "split",
    ]
    assert set(dataframe["label"]) <= {0, 1}
    assert set(written_dataframe["label"]) <= {0, 1}
    assert set(dataframe["split"]) == {"train", "val", "test"}
    assert set(written_dataframe["split"]) == {"train", "val", "test"}
    assert dataframe["split"].value_counts().to_dict() == {"train": 8, "val": 2, "test": 2}
    assert written_dataframe["split"].value_counts().to_dict() == {
        "train": 8,
        "val": 2,
        "test": 2,
    }
    assert "extra_a" not in set(dataframe["id_code"])
    assert "extra_b" not in set(dataframe["id_code"])

    merged = dataframe.sort_values("id_code").reset_index(drop=True).merge(
        pd.DataFrame(
            {
                "id_code": [Path(image_name).stem for image_name in image_names],
                "diagnosis": diagnoses,
            }
        ),
        on=["id_code", "diagnosis"],
        how="inner",
    )
    assert len(merged) == 12
    assert dataframe.loc[dataframe["diagnosis"] >= 2, "label"].eq(1).all()
    assert dataframe.loc[dataframe["diagnosis"] < 2, "label"].eq(0).all()
    assert all((image_root / relative_path).exists() for relative_path in dataframe["image_path"])


def test_prepare_metadata_rejects_missing_labeled_images(tmp_path: Path) -> None:
    labels_csv = tmp_path / "DR_grading.csv"
    image_root = tmp_path / "DR_grading"
    output_csv = tmp_path / "ddr_metadata.csv"

    image_names = [f"image_{index:02d}.jpg" for index in range(6)]
    diagnoses = [0, 1, 2, 3, 4, 2]

    _write_labels_csv(labels_csv, image_names, diagnoses)
    _touch_images(image_root, image_names[:-1])

    with pytest.raises(FileNotFoundError, match="missing under"):
        prepare_metadata(
            labels_csv=labels_csv,
            image_root=image_root,
            output_csv=output_csv,
        )


def test_prepare_metadata_can_drop_missing_labeled_rows(tmp_path: Path) -> None:
    labels_csv = tmp_path / "DR_grading.csv"
    image_root = tmp_path / "DR_grading"
    output_csv = tmp_path / "ddr_metadata.csv"

    image_names = [f"image_{index:02d}.jpg" for index in range(14)]
    diagnoses = [0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4, 2, 0, 2]

    _write_labels_csv(labels_csv, image_names, diagnoses)
    present_images = [
        image_name
        for image_name in image_names
        if image_name not in {"image_01.jpg", "image_12.jpg"}
    ]
    _touch_images(image_root, present_images)

    dataframe = prepare_metadata(
        labels_csv=labels_csv,
        image_root=image_root,
        output_csv=output_csv,
        allow_missing_images=True,
        seed=11,
    )

    assert len(dataframe) == 12
    assert "image_01" not in set(dataframe["id_code"])
    assert "image_12" not in set(dataframe["id_code"])
    assert set(dataframe["split"]) == {"train", "val", "test"}
    assert output_csv.exists()
