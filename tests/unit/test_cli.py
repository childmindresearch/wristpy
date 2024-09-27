"""Test the wristpy cli."""

import pathlib

import pytest

from wristpy.core import cli


def test_parse_arguments() -> None:
    """Test the basic running of argparse with only manadatory args."""
    args = cli.parse_arguments(["/path/to/input/file.bin"])

    assert args.input == pathlib.Path("/path/to/input/file.bin")
    assert args.output is None
    assert args.calibrator == "none"
    assert args.epoch_length == 5
    assert isinstance(args.thresholds, list)
    assert all(isinstance(threshold, float) for threshold in args.thresholds)


def test_parse_arguments_with_options() -> None:
    """Test running the argparser with an optional arg."""
    args = cli.parse_arguments(
        [
            "/path/to/input/file.bin",
            "-o",
            "/path/to/output/file.csv",
            "-c",
            "ggir",
            "-t",
            "0.1",
            "1.0",
            "1.5",
            "-e",
            "0",
        ]
    )

    assert args.input == pathlib.Path("/path/to/input/file.bin")
    assert args.output == pathlib.Path("/path/to/output/file.csv")
    assert args.calibrator == "ggir"
    assert args.thresholds == [0.1, 1.0, 1.5]
    assert args.epoch_length == 0


def test_parse_arguements_no_input() -> None:
    """Test the error when required argument is missing."""
    with pytest.raises(SystemExit):
        cli.parse_arguments([])
