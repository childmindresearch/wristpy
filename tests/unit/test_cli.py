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
            "enmo",
            "-t",
            "0.1, 1.0, 1.5",
            "-e",
            "0",
        ]
    )

    assert args.input == pathlib.Path("/path/to/input/file.bin")
    assert args.output == pathlib.Path("/path/to/output/file.csv")
    assert args.calibrator == "ggir"
    assert args.activity_metric == "enmo"
    assert args.thresholds == [0.1, 1.0, 1.5]
    assert args.epoch_length == 0


def test_parse_arguments_with_lower_case_conversion() -> None:
    """Test running the argparser lower case conversion."""
    args = cli.parse_arguments(
        [
            "/path/to/input/file.bin",
            "-o",
            "/path/to/output/file.csv",
            "-c",
            "GGIR",
            "-a",
            "ENMO",
        ]
    )

    assert args.input == pathlib.Path("/path/to/input/file.bin")
    assert args.output == pathlib.Path("/path/to/output/file.csv")
    assert args.calibrator == "ggir"
    assert args.activity_metric == "enmo"


def test_parse_arguments_with_none_threshold() -> None:
    """Test running the argparser with an optional arg."""
    args = cli.parse_arguments(
        [
            "/path/to/input/file.bin",
            "-t",
            "none",
        ]
    )

    assert args.input == pathlib.Path("/path/to/input/file.bin")
    assert args.thresholds is None


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
        activity_metric="enmo",
        epoch_length=5,
        nonwear_algorithm=["ggir"],
        verbosity=logging.WARNING,
        output_filetype=None,
    )


def test_main_enmo_default(
    mocker: pytest_mock.MockerFixture, sample_data_gt3x: pathlib.Path
) -> None:
    """Test that correct enmo default thresholds are pulled."""
    mock_run = mocker.patch.object(orchestrator, "_run_file")
    default_thresholds = (0.0563, 0.1916, 0.6958)

    cli.main([str(sample_data_gt3x), "-a", "enmo"])

    mock_run.assert_called_once_with(
        input=sample_data_gt3x,
        output=None,
        thresholds=default_thresholds,
        calibrator=None,
        activity_metric="enmo",
        epoch_length=5,
        nonwear_algorithm=["ggir"],
        verbosity=logging.WARNING,
    )


def test_main_mad_default(
    mocker: pytest_mock.MockerFixture, sample_data_gt3x: pathlib.Path
) -> None:
    """Test that correct mad default thresholds are pulled."""
    mock_run = mocker.patch.object(orchestrator, "_run_file")
    default_thresholds = (0.029, 0.338, 0.604)

    cli.main([str(sample_data_gt3x), "-a", "mad"])

    mock_run.assert_called_once_with(
        input=sample_data_gt3x,
        output=None,
        thresholds=default_thresholds,
        calibrator=None,
        activity_metric="mad",
        nonwear_algorithm=["ggir"],
        epoch_length=5,
        verbosity=logging.WARNING,
    )


def test_main_agcount_default(
    mocker: pytest_mock.MockerFixture, sample_data_gt3x: pathlib.Path
) -> None:
    """Test that correct ag_count default thresholds are pulled."""
    mock_run = mocker.patch.object(orchestrator, "_run_file")
    default_thresholds = (100, 3000, 5200)

    cli.main([str(sample_data_gt3x), "-a", "ag_count"])

    mock_run.assert_called_once_with(
        input=sample_data_gt3x,
        output=None,
        thresholds=default_thresholds,
        calibrator=None,
        activity_metric="ag_count",
        nonwear_algorithm=["ggir"],
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
            "3",
            "-a",
            "mad",
            "-nw",
            "cta, ggir",
        ]
    )

    mock_run.assert_called_once_with(
        input=sample_data_gt3x,
        output=test_output,
        thresholds=(0.1, 1.0, 1.5),
        calibrator="gradient",
        activity_metric="mad",
        nonwear_algorithm=["cta", "ggir"],
        epoch_length=3,
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
    """Test threshold parser with non-comma separted thresholds."""
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="Invalid value: word. Must be a comma-separated list or 'None'.",
    ):
        cli._none_or_float_list("word")


def test_incomplete_comma_separated_thresholds() -> None:
    """Test threshold parser with incomplete threshold list."""
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="Invalid value: 1, 2."
        "Must be a comma-separated list of exactly three numbers or 'None'.",
    ):
        cli._none_or_float_list("1, 2")


def test_invalid_nonwear_algorithm() -> None:
    """Test the nonwear algopriothm name parser wiht invalid input."""
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="Invalid algorithm: '1'. Must be one of: ggir, cta, detach.",
    ):
        cli.parse_nonwear_algorithms("1")
