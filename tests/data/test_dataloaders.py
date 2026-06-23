"""Tests for simple dataloader utilities."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from uncertainty_retfound.data.aptos import APTOSDataset
from uncertainty_retfound.data.dataloaders import create_dataloader
from uncertainty_retfound.data.transforms import build_torchvision_transform_from_config


def _write_fake_png(image_path: Path, color: tuple[int, int, int]) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (14, 12), color=color)
    image.save(image_path)


def test_create_dataloader_batches_dataset_samples(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "id_code": ["sample_a", "sample_b"],
            "label": [0, 1],
        }
    )
    image_root = tmp_path / "train_images"
    _write_fake_png(image_root / "sample_a.png", color=(255, 0, 0))
    _write_fake_png(image_root / "sample_b.png", color=(0, 255, 0))

    transform = build_torchvision_transform_from_config(
        {
            "resize": 10,
            "center_crop": 8,
            "to_tensor": True,
        }
    )
    dataset = APTOSDataset(
        metadata=metadata,
        image_root=image_root,
        transform=transform,
    )
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))

    assert isinstance(batch["image"], torch.Tensor)
    assert batch["image"].shape == (2, 3, 8, 8)
    assert isinstance(batch["label"], torch.Tensor)
    assert batch["label"].shape == (2,)
    assert batch["label"].tolist() == [0, 1]
    assert list(batch["image_path"]) == [
        str(image_root / "sample_a.png"),
        str(image_root / "sample_b.png"),
    ]
    assert list(batch["id_code"]) == ["sample_a", "sample_b"]
