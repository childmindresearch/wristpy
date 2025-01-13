"""Test the wristpy cli."""

import argparse
import logging
import pathlib

import pytest
import pytest_mock

from wristpy.core import cli, orchestrator


def test_parse_arguments() -> None:
    """Test the basic running of argparse with only manadatory args."""
    args = cli.parse_arguments(["/path/to/input/file.bin"])

    assert args.input == pathlib.Path("/path/to/input/file.bin")
    assert args.output is None
    assert args.calibrator == "none"
    assert args.epoch_length == 5
    assert args.thresholds is None


def test_parse_arguments_with_options() -> None:
    """Test running the argparser with an optional arg."""
    args = cli.parse_arguments(
        [
            "/path/to/input/file.bin",
            "-o",
            "/path/to/output/file.csv",
            "-c",
            "ggir",
            "-a",
            "ENMO",
            "-t",
            "0.1, 1.0, 1.5",
            "-e",
            "0",
        ]
    )

    assert args.input == pathlib.Path("/path/to/input/file.bin")
    assert args.output == pathlib.Path("/path/to/output/file.csv")
    assert args.calibrator == "ggir"
    assert args.activity_metric == "ENMO"
    assert args.thresholds == [0.1, 1.0, 1.5]
    assert args.epoch_length == 0


def test_parse_arguements_no_input() -> None:
    """Test the error when required argument is missing."""
    with pytest.raises(SystemExit):
        cli.parse_arguments([])


def test_main_default(
    mocker: pytest_mock.MockerFixture, sample_data_gt3x: pathlib.Path
) -> None:
    """Test cli with only necessary arguments."""
    mock_run = mocker.patch.object(orchestrator, "run")
    default_thresholds = None

    cli.main([str(sample_data_gt3x)])

    mock_run.assert_called_once_with(
        input=sample_data_gt3x,
        output=None,
        thresholds=default_thresholds,
        calibrator=None,
        activity_metric="ENMO",
        epoch_length=5,
        verbosity=logging.WARNING,
        output_filetype=None,
    )


def test_main_ENMO_default(
    mocker: pytest_mock.MockerFixture, sample_data_gt3x: pathlib.Path
) -> None:
    """Test cli with only necessary arguments."""
    mock_run = mocker.patch.object(orchestrator, "_run_file")
    default_thresholds = (0.0563, 0.1916, 0.6958)

    cli.main([str(sample_data_gt3x), "-a", "ENMO"])

    mock_run.assert_called_once_with(
        input=sample_data_gt3x,
        output=None,
        thresholds=default_thresholds,
        calibrator=None,
        activity_metric="ENMO",
        epoch_length=5,
        verbosity=logging.WARNING,
    )


def test_main_MAD_default(
    mocker: pytest_mock.MockerFixture, sample_data_gt3x: pathlib.Path
) -> None:
    """Test cli with only necessary arguments."""
    mock_run = mocker.patch.object(orchestrator, "_run_file")
    default_thresholds = (0.029, 0.338, 0.604)

    cli.main([str(sample_data_gt3x), "-a", "MAD"])

    mock_run.assert_called_once_with(
        input=sample_data_gt3x,
        output=None,
        thresholds=default_thresholds,
        calibrator=None,
        activity_metric="MAD",
        epoch_length=5,
        verbosity=logging.WARNING,
    )


def test_main_with_options(
    mocker: pytest_mock.MockerFixture,
    sample_data_gt3x: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Test cli with optional arguments."""
    test_output = tmp_path / "test.csv"
    mock_run = mocker.patch.object(orchestrator, "run")

    cli.main(
        [
            str(sample_data_gt3x),
            "--output",
            str(test_output),
            "-c",
            "gradient",
            "-t",
            "0.1, 1.0, 1.5",
            "-e",
            "0",
            "-a",
            "MAD",
        ]
    )

    mock_run.assert_called_once_with(
        input=sample_data_gt3x,
        output=test_output,
        thresholds=(0.1, 1.0, 1.5),
        calibrator="gradient",
        activity_metric="MAD",
        epoch_length=None,
        verbosity=logging.WARNING,
        output_filetype=None,
    )


def test_main_with_bad_thresholds(
    sample_data_gt3x: pathlib.Path,
) -> None:
    """Test cli with bad thresholds."""
    with pytest.raises(
        ValueError,
        match="Threshold values must be >=0, unique, and in ascending order.",
    ):
        cli.main([str(sample_data_gt3x), "-t", "10.0, 1.0, 1.5"])


def test_main_with_bad_epoch(
    sample_data_gt3x: pathlib.Path,
) -> None:
    """Test cli with invalid epoch length."""
    with pytest.raises(
        ValueError,
        match="Value for epoch_length is:-5." "Please enter an integer >= 0.",
    ):
        cli.main([str(sample_data_gt3x), "-e", "-5"])


def test_non_comma_separated_thresholds() -> None:
    """Test cli with non-comma separted thresholds."""
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="Invalid value: 1 2 3. Must be a comma-separated list or 'None'.",
    ):
        cli._none_or_float_list("1 2 3")


def test_incomplete_comma_separated_thresholds() -> None:
    """Test cli with incomplete threshold list."""
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="Invalid value: 1, 2."
        "Must be a comma-separated list of exactly three floats or 'None'.",
    ):
        cli._none_or_float_list("1, 2")
