"""Smoke tests for wristpy cli."""

import pathlib

import pytest

from wristpy.core import cli, orchestrator


def test_main_defaults(sample_data_gt3x: pathlib.Path) -> None:
    """Test cli with only necessary arguments."""
    results = cli.main([str(sample_data_gt3x)])

    assert isinstance(results, orchestrator.Results)
    assert results.enmo is not None
    assert results.anglez is not None
    assert results.nonwear_epoch is not None
    assert results.physical_activity_levels is not None
    assert results.sleep_windows_epoch is not None


def test_main_options(sample_data_gt3x: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Test cli with options."""
    options_list = [
        str(sample_data_gt3x),
        "--output",
        str(tmp_path / "test.csv"),
        "-c",
        "gradient",
        "-t",
        "0.1",
        "1.0",
        "1.5",
        "-e",
        "5",
    ]

    results = cli.main(options_list)

    assert isinstance(results, orchestrator.Results)
    assert results.enmo is not None
    assert results.anglez is not None
    assert results.nonwear_epoch is not None
    assert results.physical_activity_levels is not None
    assert results.sleep_windows_epoch is not None


def test_main_no_arguments(
    sample_data_gt3x: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Tests cli with no options."""
    with pytest.raises(SystemExit):
        cli.main([])
