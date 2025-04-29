"""Test the wristpy cli."""

import logging
import pathlib

import pytest
import pytest_mock
from typer.testing import CliRunner

from wristpy.core import cli, orchestrator


@pytest.fixture
def create_typer_cli_runner() -> CliRunner:
    """Create a Typer CLI runner."""
    return CliRunner()


def test_main_default(
    mocker: pytest_mock.MockerFixture,
    sample_data_gt3x: pathlib.Path,
    create_typer_cli_runner: CliRunner,
) -> None:
    """Test cli with only necessary arguments."""
    mock_run = mocker.patch.object(orchestrator, "run")
    default_thresholds = None

    result = create_typer_cli_runner.invoke(cli.app, [str(sample_data_gt3x)])

    assert result.exit_code == 0
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
    mocker: pytest_mock.MockerFixture,
    sample_data_gt3x: pathlib.Path,
    create_typer_cli_runner: CliRunner,
) -> None:
    """Test that correct enmo default thresholds are pulled."""
    mock_run = mocker.patch.object(orchestrator, "_run_file")
    default_thresholds = (0.0563, 0.1916, 0.6958)

    create_typer_cli_runner.invoke(cli.app, ([str(sample_data_gt3x), "-a", "enmo"]))

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
    mocker: pytest_mock.MockerFixture,
    sample_data_gt3x: pathlib.Path,
    create_typer_cli_runner: CliRunner,
) -> None:
    """Test that correct mad default thresholds are pulled."""
    mock_run = mocker.patch.object(orchestrator, "_run_file")
    default_thresholds = (0.029, 0.338, 0.604)

    create_typer_cli_runner.invoke(cli.app, ([str(sample_data_gt3x), "-a", "mad"]))

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
    mocker: pytest_mock.MockerFixture,
    sample_data_gt3x: pathlib.Path,
    create_typer_cli_runner: CliRunner,
) -> None:
    """Test that correct ag_count default thresholds are pulled."""
    mock_run = mocker.patch.object(orchestrator, "_run_file")
    default_thresholds = (100, 3000, 5200)

    create_typer_cli_runner.invoke(cli.app, ([str(sample_data_gt3x), "-a", "ag_count"]))

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
    create_typer_cli_runner: CliRunner,
) -> None:
    """Test cli with optional arguments."""
    test_output = tmp_path / "test.csv"
    mock_run = mocker.patch.object(orchestrator, "run")

    create_typer_cli_runner.invoke(
        cli.app,
        (
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
                "3",
                "-a",
                "mad",
                "-n",
                "cta",
                "-n",
                "ggir",
            ]
        ),
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
    create_typer_cli_runner: CliRunner,
) -> None:
    """Test cli with bad thresholds."""
    result = create_typer_cli_runner.invoke(
        cli.app, [str(sample_data_gt3x), "-t", "-3.0"]
    )

    assert result.exit_code != 0
    assert "Option '-t' requires 3 arguments." in result.output


def test_main_with_bad_epoch(
    sample_data_gt3x: pathlib.Path,
    create_typer_cli_runner: CliRunner,
) -> None:
    """Test cli with invalid epoch length."""
    result = create_typer_cli_runner.invoke(
        cli.app, ([str(sample_data_gt3x), "-e", "-5"])
    )

    assert result.exit_code != 0
    assert (
        "Invalid value for '-e' / '--epoch-length': -5 is not in the range x>=1."
        in result.output
    )
