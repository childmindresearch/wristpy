"""Tests the main workflow of wristpy orchestrator."""

import pathlib

import pytest

from wristpy.core import orchestrator


@pytest.fixture
def test_file() -> pathlib.Path:
    """Test data to run."""
    return (
        pathlib.Path(__file__).parent.parent / "sample_data" / "example_actigraph.gt3x"
    )


@pytest.mark.parametrize(
    "file_name", [pathlib.Path("test_output.csv"), pathlib.Path("test_output.parquet")]
)
def test_happy_path(
    file_name: pathlib.Path, tmp_path: pathlib.Path, test_file: pathlib.Path
) -> None:
    """Happy path for orchestrator."""
    results = orchestrator.run(input=test_file, output=tmp_path / file_name)

    assert (tmp_path / file_name).exists()
    assert isinstance(results, orchestrator.Results)
    assert results.enmo is not None
    assert results.anglez is not None
    assert results.nonwear_epoch is not None
    assert results.sleep_windows_epoch is not None
    assert results.physical_activity_levels is not None


def test_different_epoch(tmp_path: pathlib.Path, test_file: pathlib.Path) -> None:
    """Test using none default epoch."""
    results = orchestrator.run(
        input=test_file, output=tmp_path / "good_file.csv", epoch_length=None
    )

    assert (tmp_path / "good_file.csv").exists()
    assert isinstance(results, orchestrator.Results)
    assert results.enmo is not None
    assert results.anglez is not None
    assert results.nonwear_epoch is not None
    assert results.sleep_windows_epoch is not None
    assert results.physical_activity_levels is not None


def test_bad_calibrator(test_file: pathlib.Path) -> None:
    """Test run when invalid calibrator given."""
    with pytest.raises(
        ValueError,
        match="Invalid calibrator: Ggir. Choose: 'ggir', 'gradient'. "
        "Enter None if no calibration is desired.",
    ):
        orchestrator.run(input=test_file, calibrator="Ggir")


def test_bad_save_file(test_file: pathlib.Path) -> None:
    """Tests run when incorrect save file format given."""
    with pytest.raises(
        orchestrator.InvalidFileTypeError,
        match="The extension: .marquet is not supported."
        "Please save the file as .csv or .parquet",
    ):
        orchestrator.run(input=test_file, output=pathlib.Path("test_output.marquet"))


def test_bad_path(test_file: pathlib.Path) -> None:
    """Tests run when bad path is given."""
    with pytest.raises(orchestrator.DirectoryNotFoundError):
        orchestrator.run(
            input=test_file, output=pathlib.Path("this/path/isnt/real.csv")
        )
