"""Test the wristpy cli."""

import pathlib

import pytest
import pytest_mock

from wristpy.core import cli, config, orchestrator


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


def test_main_default(
    mocker: pytest_mock.MockerFixture, sample_data_gt3x: pathlib.Path
) -> None:
    """Test cli with only necessary arguments."""
    default_settings = config.Settings()
    mock_run = mocker.patch.object(orchestrator, "run")

    cli.main([str(sample_data_gt3x)])

    mock_run.assert_called_once_with(
        input=sample_data_gt3x,
        output=None,
        settings=default_settings,
        calibrator=None,
        epoch_length=5,
    )


def test_main_with_options(
    mocker: pytest_mock.MockerFixture,
    sample_data_gt3x: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Test cli with optional arguments."""
    test_settings = config.Settings(
        LIGHT_THRESHOLD=0.1, MODERATE_THRESHOLD=1.0, VIGOROUS_THRESHOLD=1.5
    )
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
            "0.1",
            "1.0",
            "1.5",
            "-e",
            "0",
        ]
    )

    mock_run.assert_called_once_with(
        input=sample_data_gt3x,
        output=test_output,
        settings=test_settings,
        calibrator="gradient",
        epoch_length=None,
    )
