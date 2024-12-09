"""Test the orchestrator.py module."""

import datetime
import pathlib
import re
from typing import Optional

import numpy as np
import polars as pl
import pytest

from wristpy.core import exceptions, models, orchestrator
from wristpy.processing import analytics


@pytest.fixture
def dummy_results() -> models.OrchestratorResults:
    """Makes a results object for the purpose of testing."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_measure = models.Measurement(
        measurements=np.random.rand(100),
        time=pl.Series(
            [dummy_date + datetime.timedelta(seconds=i) for i in range(100)]
        ),
    )
    dummy_results = models.OrchestratorResults(
        enmo=dummy_measure,
        anglez=dummy_measure,
        physical_activity_levels=dummy_measure,
        nonwear_epoch=dummy_measure,
        sleep_windows_epoch=dummy_measure,
    )

    return dummy_results


def test_format_sleep() -> None:
    """Test sleep formatter."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + datetime.timedelta(seconds=i) for i in range(5)]
    dummy_measure = models.Measurement(
        measurements=np.ones(5), time=pl.Series(dummy_datetime_list)
    )
    sleep_window_1 = analytics.SleepWindow(
        onset=dummy_date, wakeup=dummy_date + datetime.timedelta(seconds=1)
    )
    sleep_window_2 = analytics.SleepWindow(
        onset=dummy_date + datetime.timedelta(seconds=3),
        wakeup=dummy_date + datetime.timedelta(seconds=4),
    )

    sleep_array = orchestrator.format_sleep_data(
        sleep_windows=[sleep_window_1, sleep_window_2], reference_measure=dummy_measure
    )

    assert (
        len(sleep_array) == len(dummy_measure.measurements) == len(dummy_measure.time)
    )
    assert np.all(sleep_array == np.array([1, 1, 0, 1, 1]))


def test_format_nonwear() -> None:
    """Test nonwear formatter."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_datetime_list = [
        dummy_date + datetime.timedelta(seconds=i * 5) for i in range(4)
    ]
    dummy_epoch = models.Measurement(
        measurements=np.ones(4), time=pl.Series(dummy_datetime_list)
    )
    nonwear_measurement = models.Measurement(
        measurements=np.array([1, 0]),
        time=pl.Series([dummy_date, dummy_date + datetime.timedelta(seconds=10)]),
    )

    nonwear_epoch = orchestrator.format_nonwear_data(
        nonwear_data=nonwear_measurement,
        reference_measure=dummy_epoch,
        original_temporal_resolution=5,
    )

    assert len(nonwear_epoch) == len(dummy_epoch.measurements) == len(dummy_epoch.time)
    assert np.all(nonwear_epoch == np.array([1, 1, 0, 0]))


def test_bad_calibrator(sample_data_gt3x: pathlib.Path) -> None:
    """Test run when invalid calibrator given."""
    with pytest.raises(
        ValueError,
        match="Invalid calibrator: Ggir. Choose: 'ggir', 'gradient'. "
        "Enter None if no calibration is desired.",
    ):
        orchestrator._run_file(input=sample_data_gt3x, calibrator="Ggir")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "file_name", [pathlib.Path("test_output.csv"), pathlib.Path("test_output.parquet")]
)
def test_save_results(
    dummy_results: models.OrchestratorResults,
    file_name: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Test saving."""
    dummy_results.save_results(tmp_path / file_name)

    assert (tmp_path / file_name).exists()


def test_validate_output_invalid_file_type(tmp_path: pathlib.Path) -> None:
    """Test when a bad extention is given."""
    with pytest.raises(exceptions.InvalidFileTypeError):
        models.OrchestratorResults.validate_output(tmp_path / "bad_file.oops")


def test_run_single_file(
    sample_data_gt3x: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Testing running a single file."""
    output_file_path = tmp_path / "file_name.csv"
    results = orchestrator.run(input=sample_data_gt3x, output=output_file_path)

    assert output_file_path.exists()
    assert isinstance(results, models.OrchestratorResults)


def test_run_single_file_bad_output_filetype(
    sample_data_gt3x: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Testing running a single file."""
    output_file_path = tmp_path / "file_name.csv"

    with pytest.raises(
        ValueError,
        match="When processing single files, output_filetype should be None - "
        "the file type will be determined from the output path.",
    ):
        orchestrator.run(
            input=sample_data_gt3x, output=output_file_path, output_filetype=".csv"
        )


def test_run_dir(tmp_path: pathlib.Path, sample_data_gt3x: pathlib.Path) -> None:
    """Test run function when pointed at a directory."""
    input_dir = pathlib.Path(__file__).parent.parent / "sample_data"
    expected_files = {
        tmp_path / "example_actigraph.csv",
        tmp_path / "example_geneactiv.csv",
        tmp_path / "example_actigraph_idle_sleep_mode.csv",
    }

    results = orchestrator.run(input=input_dir, output=tmp_path, output_filetype=".csv")

    assert set(tmp_path.glob("*.csv")) == expected_files
    assert isinstance(results, dict)


def test_run_bad_dir(sample_data_gt3x: pathlib.Path) -> None:
    """Test run function when input is a directory but output is invalid."""
    input_dir = pathlib.Path(__file__).parent.parent / "sample_data"
    bad_output_dir = sample_data_gt3x

    with pytest.raises(
        ValueError,
        match="Output is a file, but must be a directory when input is a directory.",
    ):
        orchestrator.run(input=input_dir, output=bad_output_dir)


@pytest.mark.parametrize("invalid_file_type", [".zip", None])
def test_bad_file_type(
    tmp_path: pathlib.Path, invalid_file_type: Optional[str]
) -> None:
    """Test run function when output file type is invalid."""
    input_dir = pathlib.Path(__file__).parent.parent / "sample_data"
    expected_message = (
        "Invalid output_filetype: "
        f"{invalid_file_type}. Valid options are: {orchestrator.VALID_FILE_TYPES}."
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        orchestrator.run(
            input=input_dir,
            output=tmp_path,
            output_filetype=invalid_file_type,  # type: ignore[arg-type]
        )
