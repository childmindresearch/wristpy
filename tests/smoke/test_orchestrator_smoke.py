"""Tests the main workflow of wristpy orchestrator."""

import pathlib

import pytest

from wristpy.core import models, orchestrator
from wristpy.io.writers import writers


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
    assert isinstance(results, writers.OrchestratorResults)
    assert isinstance(results.anglez, models.Measurement)
    assert isinstance(results.nonwear_status, models.Measurement)
    assert isinstance(results.sleep_status, models.Measurement)
    assert all(
        isinstance(pa_metric, models.Measurement)
        for pa_metric in results.physical_activity_metric
    )
    assert all(
        isinstance(pa_level, models.Measurement)
        for pa_level in results.physical_activity_levels
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
    assert isinstance(results, writers.OrchestratorResults)
    assert isinstance(results.anglez, models.Measurement)
    assert isinstance(results.nonwear_status, models.Measurement)
    assert isinstance(results.sleep_status, models.Measurement)
    assert isinstance(results.processing_params, dict)
    assert results.processing_params["calibrator"] == "gradient"
    assert results.processing_params["nonwear_algorithm"] == ["ggir"]
    assert results.processing_params["activity_metric"] == ["enmo"]
    assert all(
        isinstance(pa_metric, models.Measurement)
        for pa_metric in results.physical_activity_metric
    )
    assert all(
        isinstance(pa_level, models.Measurement)
        for pa_level in results.physical_activity_levels
    )
