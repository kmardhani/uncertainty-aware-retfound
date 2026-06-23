"""Tests for simple APTOS image preprocessing utilities."""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from uncertainty_retfound.data.aptos import APTOSDataset
from uncertainty_retfound.data.transforms import (
    CenterCropImage,
    ComposeTransforms,
    EnsureRGB,
    ResizeImage,
    build_transform_from_config,
)


def _make_image(mode: str = "RGB", size: tuple[int, int] = (100, 80)) -> Image.Image:
    color = 128 if mode == "L" else (64, 128, 192)
    return Image.new(mode, size, color=color)


def _write_fake_png(image_path: Path, mode: str = "RGB", size: tuple[int, int] = (100, 80)) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    _make_image(mode=mode, size=size).save(image_path)


def test_ensure_rgb_converts_grayscale_image() -> None:
    image = _make_image(mode="L", size=(32, 32))

    transformed = EnsureRGB()(image)

    assert transformed.mode == "RGB"
    assert transformed.size == (32, 32)


def test_resize_image_with_integer_size() -> None:
    image = _make_image(size=(20, 10))

    transformed = ResizeImage(64)(image)

    assert transformed.size == (64, 64)


def test_resize_image_with_tuple_size() -> None:
    image = _make_image(size=(20, 10))

    transformed = ResizeImage((80, 60))(image)

    assert transformed.size == (80, 60)


def test_center_crop_image_with_integer_size() -> None:
    image = _make_image(size=(100, 80))

    transformed = CenterCropImage(32)(image)

    assert transformed.size == (32, 32)


def test_center_crop_image_with_tuple_size() -> None:
    image = _make_image(size=(100, 80))

    transformed = CenterCropImage((40, 30))(image)

    assert transformed.size == (40, 30)


def test_center_crop_image_rejects_crop_larger_than_image() -> None:
    image = _make_image(size=(24, 24))

    with pytest.raises(ValueError, match="must not exceed image size"):
        CenterCropImage(32)(image)


def test_compose_transforms_applies_transforms_in_order() -> None:
    image = _make_image(mode="L", size=(100, 80))
    transform = ComposeTransforms(
        [
            EnsureRGB(),
            ResizeImage((80, 60)),
            CenterCropImage((40, 30)),
        ]
    )

    transformed = transform(image)

    assert transformed.mode == "RGB"
    assert transformed.size == (40, 30)


def test_build_transform_from_config_creates_expected_pipeline() -> None:
    image = _make_image(mode="L", size=(100, 80))
    transform = build_transform_from_config(
        {
            "ensure_rgb": True,
            "resize": 64,
            "center_crop": 32,
        }
    )

    transformed = transform(image)

    assert transformed.mode == "RGB"
    assert transformed.size == (32, 32)


def test_dataset_transform_changes_returned_image_size(tmp_path: Path) -> None:
    metadata = pd.DataFrame({"id_code": ["sample_a"], "label": [1]})
    image_root = tmp_path / "train_images"
    _write_fake_png(image_root / "sample_a.png", mode="L", size=(100, 80))

    dataset = APTOSDataset(
        metadata=metadata,
        image_root=image_root,
        transform=build_transform_from_config(
            {
                "ensure_rgb": True,
                "resize": 64,
                "center_crop": 32,
            }
        ),
    )

    item = dataset[0]

    assert item["image"].mode == "RGB"
    assert item["image"].size == (32, 32)
