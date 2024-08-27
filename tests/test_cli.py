"""Test the wristpy cli."""

import pathlib

import pytest

from wristpy.core import cli


def test_parse_arguments() -> None:
    """Test the basic running of argparse with only manadatory args."""
    args = cli.parse_arguments(["/path/to/input/file.bin"])

    assert args.input == pathlib.Path("/path/to/input/file.bin")
    assert args.output is None
    assert isinstance(args.light_threshold, float)
    assert isinstance(args.moderate_threshold, float)
    assert isinstance(args.vigorous_threshold, float)
    assert isinstance(args.chunked, bool)
    assert isinstance(args.min_acceleration, float)
    assert isinstance(args.min_hours, int)
    assert isinstance(args.min_std, float)
    assert isinstance(args.max_iterations, int)
    assert isinstance(args.error_tolerance, float)
    assert isinstance(args.min_calibration_error, float)
    assert isinstance(args.short_length, int)
    assert isinstance(args.short_in_long, int)
    assert isinstance(args.std, float)
    assert isinstance(args.range, float)


def test_parse_arguments_with_options() -> None:
    """Test running the argparser with an optional arg."""
    args = cli.parse_arguments(
        ["/path/to/input/file.bin", "-o", "/path/to/output/file.csv"]
    )

    assert args.input == pathlib.Path("/path/to/input/file.bin")
    assert args.output == pathlib.Path("/path/to/output/file.csv")


def test_parse_arguements_no_input() -> None:
    """Test the error when required argument is missing."""
    with pytest.raises(SystemExit):
        cli.parse_arguments([])
