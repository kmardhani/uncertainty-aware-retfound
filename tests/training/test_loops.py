"""Tests for reusable training loop helpers."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from uncertainty_retfound.data.aptos import APTOSDataset
from uncertainty_retfound.data.dataloaders import create_dataloader
from uncertainty_retfound.data.transforms import build_torchvision_transform_from_config
from uncertainty_retfound.models.baseline import SmallCNNClassifier
from uncertainty_retfound.training.loops import train_one_epoch


def _write_fake_png(image_path: Path, color: tuple[int, int, int]) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (18, 18), color=color)
    image.save(image_path)


def test_train_one_epoch_updates_model_parameters(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "id_code": ["sample_a", "sample_b", "sample_c", "sample_d"],
            "label": [0, 1, 0, 1],
        }
    )
    image_root = tmp_path / "train_images"
    _write_fake_png(image_root / "sample_a.png", color=(255, 0, 0))
    _write_fake_png(image_root / "sample_b.png", color=(0, 255, 0))
    _write_fake_png(image_root / "sample_c.png", color=(0, 0, 255))
    _write_fake_png(image_root / "sample_d.png", color=(255, 255, 0))

    transform = build_torchvision_transform_from_config(
        {
            "resize": 32,
            "center_crop": 32,
            "to_tensor": True,
        }
    )
    dataset = APTOSDataset(
        metadata=metadata,
        image_root=image_root,
        transform=transform,
    )
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)
    model = SmallCNNClassifier(num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    parameter_before = model.classifier.weight.detach().clone()

    result = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
    )

    assert "loss" in result
    assert "num_batches" in result
    assert "num_examples" in result
    assert isinstance(result["loss"], float)
    assert result["loss"] >= 0.0
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert result["num_batches"] == 2
    assert result["num_examples"] == 4

    parameter_after = model.classifier.weight.detach()
    assert not torch.equal(parameter_before, parameter_after)


def test_train_one_epoch_rejects_empty_dataloader(tmp_path: Path) -> None:
    metadata = pd.DataFrame({"id_code": [], "label": []})
    dataset = APTOSDataset(
        metadata=metadata,
        image_root=tmp_path / "train_images",
        validate_paths=False,
        transform=build_torchvision_transform_from_config({"to_tensor": True}),
    )
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)
    model = SmallCNNClassifier(num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="empty dataloader"):
        train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
        )
