"""Tests for the APTOS metadata preparation script."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.data.prepare_aptos_metadata import prepare_metadata


FIXTURE_CONFIG = Path("configs/datasets/fixtures/aptos2019_sample.yaml")


def test_prepare_metadata_writes_expected_fixture_output(tmp_path: Path) -> None:
    output_path = tmp_path / "aptos_metadata.csv"

    dataframe = prepare_metadata(
        config_path=FIXTURE_CONFIG,
        output_path=output_path,
        task="referable_dr",
    )

    written_dataframe = pd.read_csv(output_path)

    assert output_path.exists()
    assert len(dataframe) == 5
    assert len(written_dataframe) == 5
    assert {"id_code", "diagnosis", "label", "task", "split"}.issubset(dataframe.columns)
    assert {"id_code", "diagnosis", "label", "task", "split"}.issubset(written_dataframe.columns)
    assert dataframe["task"].eq("referable_dr").all()
    assert written_dataframe["task"].eq("referable_dr").all()
    assert set(dataframe["label"]) <= {0, 1}
    assert set(written_dataframe["label"]) <= {0, 1}
    assert set(dataframe["split"]) <= {"train", "val", "test"}
    assert set(written_dataframe["split"]) <= {"train", "val", "test"}
    assert set(dataframe["split"]) == {"train", "val", "test"}
    assert set(written_dataframe["split"]) == {"train", "val", "test"}


def test_prepare_metadata_rejects_unknown_task(tmp_path: Path) -> None:
    output_path = tmp_path / "invalid_task.csv"

    with pytest.raises(KeyError, match="Task 'invalid_task' not found"):
        prepare_metadata(
            config_path=FIXTURE_CONFIG,
            output_path=output_path,
            task="invalid_task",
        )
