"""Tests for the RETFound feature export script."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from scripts.features.export_retfound_features import main, run_feature_export


def _write_fake_image(image_path: Path, color: tuple[int, int, int]) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), color=color)
    image.save(image_path)


def _write_fake_retfound_repo(repo_path: Path) -> None:
    repo_path.mkdir(parents=True, exist_ok=True)
    (repo_path / "models_vit.py").write_text(
        """
import torch
from torch import nn


class FakeRETFoundEncoder(nn.Module):
    def __init__(self, feature_dim: int = 8, num_classes: int = 1000) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(3, feature_dim)
        self.head = nn.Linear(feature_dim, 1000)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(inputs).flatten(1)
        return self.proj(pooled)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        return self.head(features)


def RETFound_mae(**kwargs) -> nn.Module:
    return FakeRETFoundEncoder(feature_dim=8, **kwargs)
""".strip(),
        encoding="utf-8",
    )


def _write_fake_retfound_checkpoint(repo_path: Path, checkpoint_path: Path) -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("fake_models_vit_features", repo_path / "models_vit.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load fake models_vit.py for test setup.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["fake_models_vit_features"] = module
    spec.loader.exec_module(module)

    model = module.RETFound_mae()
    state_dict = {
        f"module.{key}": value.clone() for key, value in model.state_dict().items()
    }
    torch.save({"model": state_dict}, checkpoint_path)


def _make_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_code": [
                "sample_a",
                "sample_b",
                "sample_c",
                "sample_d",
                "sample_e",
            ],
            "image_path": [
                "sample_a.png",
                "sample_b.png",
                "sample_c.png",
                "sample_d.png",
                "sample_e.png",
            ],
            "label": [0, 1, 0, 1, 1],
            "split": ["train", "train", "val", "test", "test"],
        }
    )


def test_run_feature_export_writes_expected_outputs(tmp_path: Path) -> None:
    metadata = _make_metadata()
    metadata_path = tmp_path / "prepared_metadata.csv"
    image_root = tmp_path / "images"
    output_dir = tmp_path / "features"
    repo_path = tmp_path / "fake_retfound_repo"
    checkpoint_path = tmp_path / "retfound_checkpoint.pth"

    _write_fake_image(image_root / "sample_a.png", color=(255, 0, 0))
    _write_fake_image(image_root / "sample_b.png", color=(0, 255, 0))
    _write_fake_image(image_root / "sample_c.png", color=(0, 0, 255))
    _write_fake_image(image_root / "sample_d.png", color=(255, 255, 0))
    _write_fake_image(image_root / "sample_e.png", color=(0, 255, 255))
    metadata.to_csv(metadata_path, index=False)
    _write_fake_retfound_repo(repo_path)
    _write_fake_retfound_checkpoint(repo_path, checkpoint_path)

    result = run_feature_export(
        metadata_csv=metadata_path,
        image_root=image_root,
        output_dir=output_dir,
        backbone_checkpoint=checkpoint_path,
        retfound_repo_path=repo_path,
        batch_size=2,
        resize=32,
        center_crop=32,
        num_workers=0,
        device="cpu",
        feature_dim=8,
        show_progress=False,
    )

    train_features = torch.load(output_dir / "train_features.pt", map_location="cpu")
    val_features = torch.load(output_dir / "val_features.pt", map_location="cpu")
    test_features = torch.load(output_dir / "test_features.pt", map_location="cpu")
    metadata_json = json.loads((output_dir / "feature_metadata.json").read_text(encoding="utf-8"))

    assert result["counts"] == {"train": 2, "val": 1, "test": 2}
    assert train_features["split"] == "train"
    assert val_features["split"] == "val"
    assert test_features["split"] == "test"
    assert train_features["features"].shape == (2, 8)
    assert val_features["features"].shape == (1, 8)
    assert test_features["features"].shape == (2, 8)
    assert train_features["labels"].shape == (2,)
    assert val_features["labels"].shape == (1,)
    assert test_features["labels"].shape == (2,)
    assert train_features["id_codes"] == ["sample_a", "sample_b"]
    assert val_features["id_codes"] == ["sample_c"]
    assert test_features["id_codes"] == ["sample_d", "sample_e"]
    assert train_features["image_paths"] == [
        str(image_root / "sample_a.png"),
        str(image_root / "sample_b.png"),
    ]
    assert val_features["image_paths"] == [str(image_root / "sample_c.png")]
    assert test_features["image_paths"] == [
        str(image_root / "sample_d.png"),
        str(image_root / "sample_e.png"),
    ]
    assert metadata_json["model"] == "RETFound_mae"
    assert metadata_json["backbone_checkpoint"] == str(checkpoint_path)
    assert metadata_json["retfound_repo_path"] == str(repo_path)
    assert metadata_json["feature_dim"] == 8
    assert metadata_json["resize"] == 32
    assert metadata_json["center_crop"] == 32
    assert metadata_json["counts"] == {"train": 2, "val": 1, "test": 2}
    assert "timestamp" in metadata_json


def test_run_feature_export_requires_all_splits(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "id_code": ["sample_a", "sample_b"],
            "image_path": ["sample_a.png", "sample_b.png"],
            "label": [0, 1],
            "split": ["train", "val"],
        }
    )
    metadata_path = tmp_path / "prepared_metadata.csv"
    image_root = tmp_path / "images"
    repo_path = tmp_path / "fake_retfound_repo"
    checkpoint_path = tmp_path / "retfound_checkpoint.pth"

    _write_fake_image(image_root / "sample_a.png", color=(255, 0, 0))
    _write_fake_image(image_root / "sample_b.png", color=(0, 255, 0))
    metadata.to_csv(metadata_path, index=False)
    _write_fake_retfound_repo(repo_path)
    _write_fake_retfound_checkpoint(repo_path, checkpoint_path)

    with pytest.raises(ValueError, match="No rows found for split 'test'"):
        run_feature_export(
            metadata_csv=metadata_path,
            image_root=image_root,
            output_dir=tmp_path / "features",
            backbone_checkpoint=checkpoint_path,
            retfound_repo_path=repo_path,
            batch_size=2,
            resize=32,
            center_crop=32,
            num_workers=0,
            device="cpu",
            feature_dim=8,
            show_progress=False,
        )


def test_main_prints_feature_metadata_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = _make_metadata()
    metadata_path = tmp_path / "prepared_metadata.csv"
    image_root = tmp_path / "images"
    output_dir = tmp_path / "features"
    repo_path = tmp_path / "fake_retfound_repo"
    checkpoint_path = tmp_path / "retfound_checkpoint.pth"

    _write_fake_image(image_root / "sample_a.png", color=(255, 0, 0))
    _write_fake_image(image_root / "sample_b.png", color=(0, 255, 0))
    _write_fake_image(image_root / "sample_c.png", color=(0, 0, 255))
    _write_fake_image(image_root / "sample_d.png", color=(255, 255, 0))
    _write_fake_image(image_root / "sample_e.png", color=(0, 255, 255))
    metadata.to_csv(metadata_path, index=False)
    _write_fake_retfound_repo(repo_path)
    _write_fake_retfound_checkpoint(repo_path, checkpoint_path)

    main(
        [
            "--metadata-csv",
            str(metadata_path),
            "--image-root",
            str(image_root),
            "--output-dir",
            str(output_dir),
            "--backbone-checkpoint",
            str(checkpoint_path),
            "--retfound-repo-path",
            str(repo_path),
            "--batch-size",
            "2",
            "--resize",
            "32",
            "--center-crop",
            "32",
            "--feature-dim",
            "8",
            "--no-progress",
        ]
    )

    captured = capsys.readouterr()
    assert "Saved feature metadata to:" in captured.out


def test_run_feature_export_resolves_exact_png_image_paths(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "id_code": ["sample_a", "sample_b", "sample_c"],
            "image_path": ["nested/sample_a.png", "nested/sample_b.png", "nested/sample_c.png"],
            "label": [0, 1, 0],
            "split": ["train", "val", "test"],
        }
    )
    metadata_path = tmp_path / "prepared_metadata.csv"
    image_root = tmp_path / "images"
    output_dir = tmp_path / "features"
    repo_path = tmp_path / "fake_retfound_repo"
    checkpoint_path = tmp_path / "retfound_checkpoint.pth"

    _write_fake_image(image_root / "nested" / "sample_a.png", color=(255, 0, 0))
    _write_fake_image(image_root / "nested" / "sample_b.png", color=(0, 255, 0))
    _write_fake_image(image_root / "nested" / "sample_c.png", color=(0, 0, 255))
    metadata.to_csv(metadata_path, index=False)
    _write_fake_retfound_repo(repo_path)
    _write_fake_retfound_checkpoint(repo_path, checkpoint_path)

    run_feature_export(
        metadata_csv=metadata_path,
        image_root=image_root,
        output_dir=output_dir,
        backbone_checkpoint=checkpoint_path,
        retfound_repo_path=repo_path,
        batch_size=1,
        resize=32,
        center_crop=32,
        num_workers=0,
        device="cpu",
        feature_dim=8,
        show_progress=False,
    )

    train_features = torch.load(output_dir / "train_features.pt", map_location="cpu")
    assert train_features["image_paths"] == [str(image_root / "nested" / "sample_a.png")]


def test_run_feature_export_resolves_exact_jpg_image_paths(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "id_code": ["sample_a", "sample_b", "sample_c"],
            "image_path": ["sample_a.jpg", "sample_b.jpg", "sample_c.jpg"],
            "label": [0, 1, 0],
            "split": ["train", "val", "test"],
        }
    )
    metadata_path = tmp_path / "prepared_metadata.csv"
    image_root = tmp_path / "images"
    output_dir = tmp_path / "features"
    repo_path = tmp_path / "fake_retfound_repo"
    checkpoint_path = tmp_path / "retfound_checkpoint.pth"

    _write_fake_image(image_root / "sample_a.jpg", color=(255, 0, 0))
    _write_fake_image(image_root / "sample_b.jpg", color=(0, 255, 0))
    _write_fake_image(image_root / "sample_c.jpg", color=(0, 0, 255))
    metadata.to_csv(metadata_path, index=False)
    _write_fake_retfound_repo(repo_path)
    _write_fake_retfound_checkpoint(repo_path, checkpoint_path)

    run_feature_export(
        metadata_csv=metadata_path,
        image_root=image_root,
        output_dir=output_dir,
        backbone_checkpoint=checkpoint_path,
        retfound_repo_path=repo_path,
        batch_size=1,
        resize=32,
        center_crop=32,
        num_workers=0,
        device="cpu",
        feature_dim=8,
        show_progress=False,
    )

    train_features = torch.load(output_dir / "train_features.pt", map_location="cpu")
    assert train_features["image_paths"] == [str(image_root / "sample_a.jpg")]
