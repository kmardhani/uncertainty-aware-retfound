"""Tests for the APTOS dataset wrapper."""

from pathlib import Path

from PIL import Image
import pandas as pd
import pytest

from uncertainty_retfound.data.aptos import APTOSDataset


def _write_fake_png(image_path: Path, color: tuple[int, int, int]) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (4, 4), color=color)
    image.save(image_path)


def _make_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_code": ["sample_a", "sample_b"],
            "label": [0, 1],
        }
    )


def test_aptos_dataset_loads_from_dataframe_and_returns_expected_item(tmp_path: Path) -> None:
    metadata = _make_metadata()
    image_root = tmp_path / "train_images"
    _write_fake_png(image_root / "sample_a.png", color=(255, 0, 0))
    _write_fake_png(image_root / "sample_b.png", color=(0, 255, 0))

    dataset = APTOSDataset(metadata=metadata, image_root=image_root)

    item = dataset[0]

    assert len(dataset) == 2
    assert set(item) == {"image", "label", "image_path", "id_code"}
    assert item["image"].mode == "RGB"
    assert item["label"] == 0
    assert item["image_path"] == str(image_root / "sample_a.png")
    assert item["id_code"] == "sample_a"


def test_aptos_dataset_applies_transform_when_provided(tmp_path: Path) -> None:
    metadata = _make_metadata()
    image_root = tmp_path / "train_images"
    _write_fake_png(image_root / "sample_a.png", color=(255, 0, 0))
    _write_fake_png(image_root / "sample_b.png", color=(0, 255, 0))

    dataset = APTOSDataset(
        metadata=metadata,
        image_root=image_root,
        transform=lambda image: (image.mode, image.size),
    )

    item = dataset[1]

    assert item["image"] == ("RGB", (4, 4))
    assert item["label"] == 1
    assert item["id_code"] == "sample_b"


def test_aptos_dataset_loads_metadata_from_csv_path(tmp_path: Path) -> None:
    metadata = _make_metadata()
    metadata_path = tmp_path / "prepared_metadata.csv"
    image_root = tmp_path / "train_images"
    _write_fake_png(image_root / "sample_a.png", color=(255, 0, 0))
    _write_fake_png(image_root / "sample_b.png", color=(0, 255, 0))
    metadata.to_csv(metadata_path, index=False)

    dataset = APTOSDataset(metadata=metadata_path, image_root=image_root)

    assert len(dataset) == 2
    assert dataset[1]["image_path"] == str(image_root / "sample_b.png")


def test_aptos_dataset_raises_on_missing_paths_when_validation_enabled(tmp_path: Path) -> None:
    metadata = _make_metadata()
    image_root = tmp_path / "train_images"
    _write_fake_png(image_root / "sample_a.png", color=(255, 0, 0))

    with pytest.raises(FileNotFoundError, match="missing image files"):
        APTOSDataset(metadata=metadata, image_root=image_root, validate_paths=True)


def test_aptos_dataset_allows_missing_paths_until_access_when_validation_disabled(
    tmp_path: Path,
) -> None:
    metadata = _make_metadata()
    image_root = tmp_path / "train_images"
    _write_fake_png(image_root / "sample_a.png", color=(255, 0, 0))

    dataset = APTOSDataset(metadata=metadata, image_root=image_root, validate_paths=False)

    assert dataset[0]["id_code"] == "sample_a"

    with pytest.raises(FileNotFoundError):
        _ = dataset[1]
