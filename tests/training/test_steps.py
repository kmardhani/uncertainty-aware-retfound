"""Tests for small training-step helpers."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from uncertainty_retfound.data.aptos import APTOSDataset
from uncertainty_retfound.data.dataloaders import create_dataloader
from uncertainty_retfound.data.transforms import build_torchvision_transform_from_config
from uncertainty_retfound.models.baseline import SmallCNNClassifier
from uncertainty_retfound.training.steps import train_one_batch


def _write_fake_png(image_path: Path, color: tuple[int, int, int]) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (18, 18), color=color)
    image.save(image_path)


def test_train_one_batch_updates_model_parameters(tmp_path: Path) -> None:
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

    batch = next(iter(dataloader))
    parameter_before = model.classifier.weight.detach().clone()

    result = train_one_batch(model=model, batch=batch, optimizer=optimizer)

    assert "loss" in result
    assert "batch_size" in result
    assert result["batch_size"] == 2
    assert isinstance(result["loss"], float)
    assert result["loss"] >= 0.0
    assert torch.isfinite(torch.tensor(result["loss"]))

    parameter_after = model.classifier.weight.detach()
    assert not torch.equal(parameter_before, parameter_after)
