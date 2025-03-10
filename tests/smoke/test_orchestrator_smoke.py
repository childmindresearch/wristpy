"""Tests the main workflow of wristpy orchestrator."""

import pathlib

import pytest

from wristpy.core import models, orchestrator


@pytest.mark.parametrize(
    "file_name", [pathlib.Path("test_output.csv"), pathlib.Path("test_output.parquet")]
)
def test_orchestrator_happy_path(
    file_name: pathlib.Path,
    tmp_path: pathlib.Path,
    sample_data_gt3x: pathlib.Path,
) -> None:
    """Happy path for orchestrator."""
    output_path = tmp_path / file_name

    results = orchestrator._run_file(input=sample_data_gt3x, output=output_path)

    assert output_path.exists()
    assert isinstance(results, models.OrchestratorResults)
    assert isinstance(results.physical_activity_metric, models.Measurement)
    assert isinstance(results.anglez, models.Measurement)
    assert isinstance(results.nonwear_epoch, models.Measurement)
    assert isinstance(results.sleep_windows_epoch, models.Measurement)
    assert isinstance(results.physical_activity_levels, models.Measurement)


def test_orchestrator_different_epoch(
    tmp_path: pathlib.Path,
    sample_data_gt3x: pathlib.Path,
) -> None:
    """Test using none default epoch."""
    output_path = tmp_path / "good_file.csv"

    results = orchestrator._run_file(
        input=sample_data_gt3x, output=output_path, epoch_length=None
    )

    assert output_path.exists()
    assert isinstance(results, models.OrchestratorResults)
    assert isinstance(results.physical_activity_metric, models.Measurement)
    assert isinstance(results.anglez, models.Measurement)
    assert isinstance(results.nonwear_epoch, models.Measurement)
    assert isinstance(results.sleep_windows_epoch, models.Measurement)
    assert isinstance(results.physical_activity_levels, models.Measurement)


def test_orchestrator_none_epoch_ag_count(
    tmp_path: pathlib.Path,
    sample_data_gt3x: pathlib.Path,
) -> None:
    """Test using none default epoch."""
    output_path = tmp_path / "good_file.csv"

    with pytest.raises(
        ValueError,
        match="If using 'ag_count' or 'mad', epoch_length must be provided.",
    ):
        orchestrator._run_file(
            input=sample_data_gt3x,
            output=output_path,
            activity_metric="ag_count",
            epoch_length=None,
        )


def test_orchestrator_idle_sleep_mode_run(
    tmp_path: pathlib.Path,
    sample_data_gt3x_idle_sleep_mode: pathlib.Path,
) -> None:
    """Idle sleep mode path for orchestrator."""
    results = orchestrator.run(
        input=sample_data_gt3x_idle_sleep_mode, output=tmp_path / "good_file.csv"
    )

    assert (tmp_path / "good_file.csv").exists()
    assert isinstance(results, models.OrchestratorResults)
    assert isinstance(results.physical_activity_metric, models.Measurement)
    assert isinstance(results.anglez, models.Measurement)
    assert isinstance(results.nonwear_epoch, models.Measurement)
    assert isinstance(results.sleep_windows_epoch, models.Measurement)
    assert isinstance(results.physical_activity_levels, models.Measurement)
