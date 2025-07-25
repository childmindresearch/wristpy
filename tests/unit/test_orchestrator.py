"""Test the orchestrator.py module."""

import datetime
import pathlib
import re
from typing import Literal, Optional, Sequence

import numpy as np
import polars as pl
import pytest

from wristpy.core import exceptions, models, orchestrator
from wristpy.io.writers import writers


@pytest.fixture
def dummy_results() -> writers.OrchestratorResults:
    """Makes a results object for the purpose of testing."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_measure = models.Measurement(
        measurements=np.random.rand(100),
        time=pl.Series(
            [dummy_date + datetime.timedelta(seconds=i) for i in range(100)]
        ),
    )
    dummy_results = writers.OrchestratorResults(
        physical_activity_metric=[dummy_measure],
        anglez=dummy_measure,
        physical_activity_levels=[dummy_measure],
        nonwear_status=dummy_measure,
        sleep_status=dummy_measure,
        sib_periods=dummy_measure,
        spt_periods=dummy_measure,
    )

    return dummy_results


def test_bad_calibrator(sample_data_gt3x: pathlib.Path) -> None:
    """Test run when invalid calibrator given."""
    with pytest.raises(
        ValueError,
        match="Invalid calibrator: Ggir. Choose: 'ggir', 'gradient'. "
        "Enter None if no calibration is desired.",
    ):
        orchestrator._run_file(input=sample_data_gt3x, calibrator="Ggir")  # type: ignore[arg-type]


def test_bad_nonwear(sample_data_gt3x: pathlib.Path) -> None:
    """Test run when invalid calibrator given."""
    with pytest.raises(
        ValueError,
        match="Temperature data is required for the CTA and DETACH nonwear algorithms.",
    ):
        orchestrator._run_file(input=sample_data_gt3x, nonwear_algorithm=["detach"])


def test_bad_epoch_length(sample_data_gt3x: pathlib.Path) -> None:
    """Test run when invalid epoch length given."""
    with pytest.raises(
        ValueError,
        match="Epoch_length must be greater than 0.",
    ):
        orchestrator._run_file(input=sample_data_gt3x, epoch_length=-5)


@pytest.mark.parametrize(
    "file_name", [pathlib.Path("test_output.csv"), pathlib.Path("test_output.parquet")]
)
def test_save_results(
    dummy_results: writers.OrchestratorResults,
    file_name: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Test saving."""
    dummy_results.save_results(tmp_path / file_name)

    assert (tmp_path / file_name).exists()


def test_validate_output_invalid_file_type(tmp_path: pathlib.Path) -> None:
    """Test when a bad extention is given."""
    with pytest.raises(exceptions.InvalidFileTypeError):
        writers.OrchestratorResults.validate_output(tmp_path / "bad_file.oops")


def test_run_single_file(
    sample_data_bin: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Testing running a single file."""
    output_file_path = tmp_path / "file_name.csv"
    results = orchestrator.run(
        input=sample_data_bin,
        output=output_file_path,
        activity_metric=["mad"],
        calibrator="ggir",
    )

    assert output_file_path.exists()
    assert isinstance(results, writers.OrchestratorResults)


def test_run_single_file_agcount_default(
    sample_data_bin: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Testing running a single file."""
    output_file_path = tmp_path / "file_name.csv"
    results = orchestrator.run(
        input=sample_data_bin,
        output=output_file_path,
        activity_metric=["ag_count"],
        nonwear_algorithm=["detach"],
    )

    assert output_file_path.exists()
    assert isinstance(results, writers.OrchestratorResults)


@pytest.mark.parametrize("nonwear_algorithm", [["detach"], ["cta", "ggir"]])
def test_run_single_file_nonwear_options(
    sample_data_bin: pathlib.Path,
    tmp_path: pathlib.Path,
    nonwear_algorithm: Sequence[Literal["ggir", "cta", "detach"]],
) -> None:
    """Testing running a single file."""
    output_file_path = tmp_path / "file_name.csv"
    results = orchestrator.run(
        input=sample_data_bin,
        output=output_file_path,
        nonwear_algorithm=nonwear_algorithm,
    )

    assert output_file_path.exists()
    assert isinstance(results, writers.OrchestratorResults)


def test_run_dir(tmp_path: pathlib.Path) -> None:
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


def test_run_single_file_mims(
    sample_data_bin: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Testing running a single file with mims."""
    output_file_path = tmp_path / "file_name.csv"
    results = orchestrator.run(
        input=sample_data_bin,
        output=output_file_path,
        activity_metric=["mims"],
    )

    assert output_file_path.exists()
    assert isinstance(results, writers.OrchestratorResults)
