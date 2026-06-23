"""Tests for APTOS image path resolution and validation."""

from pathlib import Path

from uncertainty_retfound.data.aptos import (
    get_aptos_image_extension,
    load_aptos_metadata,
    resolve_aptos_image_paths,
    validate_aptos_image_paths,
)
from uncertainty_retfound.utils.config import load_yaml_config


FIXTURE_CONFIG = "configs/datasets/fixtures/aptos2019_sample.yaml"


def _write_fake_pngs(image_root: Path, image_ids: list[str]) -> None:
    image_root.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        (image_root / f"{image_id}.png").write_bytes(b"not-a-real-png")


def test_resolve_aptos_image_paths_uses_fixture_ids_and_png_extension(tmp_path: Path) -> None:
    dataset_config = load_yaml_config(FIXTURE_CONFIG)
    metadata = load_aptos_metadata(dataset_config, task_name="referable_dr")
    image_root = tmp_path / "train_images"

    resolved_metadata = resolve_aptos_image_paths(
        metadata,
        image_root=image_root,
        extension=get_aptos_image_extension(dataset_config),
    )

    assert resolved_metadata["image_path"].tolist() == [
        str(image_root / "sample_no_dr.png"),
        str(image_root / "sample_mild.png"),
        str(image_root / "sample_moderate.png"),
        str(image_root / "sample_severe.png"),
        str(image_root / "sample_proliferative.png"),
    ]


def test_validate_aptos_image_paths_succeeds_without_loading_pixels(tmp_path: Path) -> None:
    dataset_config = load_yaml_config(FIXTURE_CONFIG)
    metadata = load_aptos_metadata(dataset_config, task_name="referable_dr")
    image_root = tmp_path / "train_images"

    _write_fake_pngs(image_root, metadata["image_id"].tolist())

    summary = validate_aptos_image_paths(
        metadata,
        image_root=image_root,
        extension=get_aptos_image_extension(dataset_config),
    )

    assert summary.total_rows == 5
    assert summary.found_image_count == 5
    assert summary.missing_image_count == 0
    assert summary.missing_image_ids == []
    assert summary.missing_image_paths == []
    assert summary.all_images_found is True


def test_validate_aptos_image_paths_reports_missing_files(tmp_path: Path) -> None:
    dataset_config = load_yaml_config(FIXTURE_CONFIG)
    metadata = load_aptos_metadata(dataset_config, task_name="referable_dr")
    image_root = tmp_path / "train_images"

    _write_fake_pngs(
        image_root,
        ["sample_no_dr", "sample_mild", "sample_moderate", "sample_severe"],
    )

    summary = validate_aptos_image_paths(
        metadata,
        image_root=image_root,
        extension=get_aptos_image_extension(dataset_config),
        max_missing_to_show=1,
    )

    assert summary.total_rows == 5
    assert summary.found_image_count == 4
    assert summary.missing_image_count == 1
    assert summary.missing_image_ids == ["sample_proliferative"]
    assert summary.missing_image_paths == [str(image_root / "sample_proliferative.png")]
    assert summary.all_images_found is False
