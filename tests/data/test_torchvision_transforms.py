"""Tests for torchvision-based image preprocessing helpers."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from uncertainty_retfound.data.aptos import APTOSDataset
from uncertainty_retfound.data.transforms import build_torchvision_transform_from_config


def _write_fake_png(image_path: Path, color: tuple[int, int, int]) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (12, 10), color=color)
    image.save(image_path)


def test_torchvision_transform_returns_tensor_sample(tmp_path: Path) -> None:
    metadata = pd.DataFrame({"id_code": ["sample_a"], "label": [1]})
    image_root = tmp_path / "train_images"
    _write_fake_png(image_root / "sample_a.png", color=(255, 0, 0))

    transform = build_torchvision_transform_from_config(
        {
            "resize": 8,
            "center_crop": 6,
            "to_tensor": True,
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }
    )

    dataset = APTOSDataset(
        metadata=metadata,
        image_root=image_root,
        transform=transform,
    )

    sample = dataset[0]

    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 6, 6)
    assert sample["label"] == 1
    assert sample["image_path"] == str(image_root / "sample_a.png")
    assert sample["id_code"] == "sample_a"
