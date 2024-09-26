"""Tests the main workflow of wristpy orchestrator."""

import pathlib

import pytest

from wristpy.core import models, orchestrator


@pytest.mark.parametrize(
    "file_name", [pathlib.Path("test_output.csv"), pathlib.Path("test_output.parquet")]
)
def test_orchestrator_happy_path(
    file_name: pathlib.Path, tmp_path: pathlib.Path, sample_data_gt3x: pathlib.Path
) -> None:
    """Happy path for orchestrator."""
    results = orchestrator.run(input=sample_data_gt3x, output=tmp_path / file_name)

    assert (tmp_path / file_name).exists()
    assert isinstance(results, orchestrator.Results)
    assert isinstance(results.enmo, models.Measurement)
    assert isinstance(results.anglez, models.Measurement)
    assert isinstance(results.nonwear_epoch, models.Measurement)
    assert isinstance(results.sleep_windows_epoch, models.Measurement)
    assert isinstance(results.physical_activity_levels, models.Measurement)


def test_orchestrator_different_epoch(
    tmp_path: pathlib.Path, sample_data_gt3x: pathlib.Path
) -> None:
    """Test using none default epoch."""
    results = orchestrator.run(
        input=sample_data_gt3x, output=tmp_path / "good_file.csv", epoch_length=None
    )

    assert (tmp_path / "good_file.csv").exists()
    assert isinstance(results, orchestrator.Results)
    assert isinstance(results.enmo, models.Measurement)
    assert isinstance(results.anglez, models.Measurement)
    assert isinstance(results.nonwear_epoch, models.Measurement)
    assert isinstance(results.sleep_windows_epoch, models.Measurement)
    assert isinstance(results.physical_activity_levels, models.Measurement)
