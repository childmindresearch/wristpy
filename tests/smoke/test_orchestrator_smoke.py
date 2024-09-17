"""Tests the main workflow of wristpy orchestrator."""

import pathlib

import pytest

from wristpy.core import exceptions, orchestrator


@pytest.mark.parametrize(
    "file_name", [pathlib.Path("test_output.csv"), pathlib.Path("test_output.parquet")]
)
def test_happy_path(
    file_name: pathlib.Path, tmp_path: pathlib.Path, sample_data_gt3x: pathlib.Path
) -> None:
    """Happy path for orchestrator."""
    results = orchestrator.run(input=sample_data_gt3x, output=tmp_path / file_name)

    assert (tmp_path / file_name).exists()
    assert isinstance(results, orchestrator.Results)
    assert results.enmo is not None
    assert results.anglez is not None
    assert results.nonwear_epoch is not None
    assert results.sleep_windows_epoch is not None
    assert results.physical_activity_levels is not None


def test_different_epoch(
    tmp_path: pathlib.Path, sample_data_gt3x: pathlib.Path
) -> None:
    """Test using none default epoch."""
    results = orchestrator.run(
        input=sample_data_gt3x, output=tmp_path / "good_file.csv", epoch_length=None
    )

    assert (tmp_path / "good_file.csv").exists()
    assert isinstance(results, orchestrator.Results)
    assert results.enmo is not None
    assert results.anglez is not None
    assert results.nonwear_epoch is not None
    assert results.sleep_windows_epoch is not None
    assert results.physical_activity_levels is not None


def test_bad_calibrator(sample_data_gt3x: pathlib.Path) -> None:
    """Test run when invalid calibrator given."""
    with pytest.raises(
        ValueError,
        match="Invalid calibrator: Ggir. Choose: 'ggir', 'gradient'. "
        "Enter None if no calibration is desired.",
    ):
        orchestrator.run(input=sample_data_gt3x, calibrator="Ggir")  # type: ignore[arg-type]


def test_bad_save_file(sample_data_gt3x: pathlib.Path) -> None:
    """Tests run when incorrect save file format given."""
    with pytest.raises(
        exceptions.InvalidFileTypeError,
        match="The extension: .marquet is not supported."
        "Please save the file as .csv or .parquet",
    ):
        orchestrator.run(
            input=sample_data_gt3x, output=pathlib.Path("test_output.marquet")
        )


def test_bad_path(sample_data_gt3x: pathlib.Path) -> None:
    """Tests run when bad path is given."""
    with pytest.raises(exceptions.DirectoryNotFoundError):
        orchestrator.run(
            input=sample_data_gt3x, output=pathlib.Path("this/path/isnt/real.csv")
        )
