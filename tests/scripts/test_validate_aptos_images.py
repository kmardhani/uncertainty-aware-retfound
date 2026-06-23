"""Tests for the APTOS image validation script."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.data.validate_aptos_images import main, validate_aptos_images
from uncertainty_retfound.utils.config import load_yaml_config


FIXTURE_CONFIG = Path("configs/datasets/fixtures/aptos2019_sample.yaml")
FIXTURE_LABELS_CSV = Path("tests/fixtures/aptos2019/train.csv")


def _write_fake_pngs(image_root: Path, image_ids: list[str]) -> None:
    image_root.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        (image_root / f"{image_id}.png").write_bytes(b"not-a-real-png")


def _write_temp_config(tmp_path: Path, image_root: Path) -> Path:
    config = load_yaml_config(FIXTURE_CONFIG)
    config["paths"]["labels_csv"] = str(FIXTURE_LABELS_CSV)
    config["paths"]["image_dir"] = str(image_root)

    config_path = tmp_path / "aptos_fixture.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return config_path


def test_validate_aptos_images_prepares_fixture_metadata_in_memory(tmp_path: Path) -> None:
    image_root = tmp_path / "train_images"
    _write_fake_pngs(
        image_root,
        [
            "sample_no_dr",
            "sample_mild",
            "sample_moderate",
            "sample_severe",
            "sample_proliferative",
        ],
    )
    config_path = _write_temp_config(tmp_path, image_root)

    summary = validate_aptos_images(config_path=config_path, task="referable_dr")

    assert summary.total_rows == 5
    assert summary.found_image_count == 5
    assert summary.missing_image_count == 0
    assert summary.id_column == "id_code"


def test_validate_aptos_images_accepts_prepared_metadata_csv(tmp_path: Path) -> None:
    image_root = tmp_path / "train_images"
    _write_fake_pngs(image_root, ["sample_no_dr", "sample_mild"])
    config_path = _write_temp_config(tmp_path, image_root)

    metadata_csv = tmp_path / "prepared_metadata.csv"
    pd.DataFrame(
        {
            "id_code": ["sample_no_dr", "sample_mild"],
            "diagnosis": [0, 1],
            "label": [0, 0],
            "task": ["referable_dr", "referable_dr"],
            "split": ["train", "val"],
        }
    ).to_csv(metadata_csv, index=False)

    summary = validate_aptos_images(config_path=config_path, metadata_csv=metadata_csv)

    assert summary.total_rows == 2
    assert summary.found_image_count == 2
    assert summary.missing_image_count == 0


def test_main_returns_non_zero_when_images_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_root = tmp_path / "train_images"
    _write_fake_pngs(
        image_root,
        ["sample_no_dr", "sample_mild", "sample_moderate", "sample_severe"],
    )
    config_path = _write_temp_config(tmp_path, image_root)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_aptos_images.py",
            "--config",
            str(config_path),
            "--task",
            "referable_dr",
            "--max-missing-to-show",
            "1",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Missing images: 1" in captured.out
    assert "sample_proliferative" in captured.out
