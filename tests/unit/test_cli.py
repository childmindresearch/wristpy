"""Test the wristpy cli."""

import logging
import pathlib

import pytest
import pytest_mock
from typer import testing

from wristpy.core import cli, orchestrator


@pytest.fixture
def create_typer_cli_runner() -> testing.CliRunner:
    """Create a Typer CLI runner."""
    return testing.CliRunner()


def test_main_default(
    mocker: pytest_mock.MockerFixture,
    sample_data_gt3x: pathlib.Path,
    create_typer_cli_runner: testing.CliRunner,
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
        verbosity=logging.INFO,
        output_filetype=".csv",
    )


def test_main_enmo_default(
    mocker: pytest_mock.MockerFixture,
    sample_data_gt3x: pathlib.Path,
    create_typer_cli_runner: testing.CliRunner,
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
        verbosity=logging.INFO,
    )


def test_main_mad_default(
    mocker: pytest_mock.MockerFixture,
    sample_data_gt3x: pathlib.Path,
    create_typer_cli_runner: testing.CliRunner,
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
        verbosity=logging.INFO,
    )


def test_main_agcount_default(
    mocker: pytest_mock.MockerFixture,
    sample_data_gt3x: pathlib.Path,
    create_typer_cli_runner: testing.CliRunner,
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
        verbosity=logging.INFO,
    )


def test_main_with_options(
    mocker: pytest_mock.MockerFixture,
    sample_data_gt3x: pathlib.Path,
    tmp_path: pathlib.Path,
    create_typer_cli_runner: testing.CliRunner,
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
        verbosity=logging.INFO,
        output_filetype=".csv",
    )


def test_main_with_bad_thresholds(
    sample_data_gt3x: pathlib.Path,
    create_typer_cli_runner: testing.CliRunner,
) -> None:
    """Test cli with bad thresholds."""
    result = create_typer_cli_runner.invoke(
        cli.app, [str(sample_data_gt3x), "-t", "3.0"]
    )

    assert result.exit_code != 0
    # partial matching due to ANSI escape sequences in Github Actions
    assert "requires 3 arguments." in result.output


def test_main_with_bad_epoch(
    sample_data_gt3x: pathlib.Path,
    create_typer_cli_runner: testing.CliRunner,
) -> None:
    """Test cli with invalid epoch length."""
    result = create_typer_cli_runner.invoke(
        cli.app, [str(sample_data_gt3x), "-e", "-5"]
    )

    assert result.exit_code != 0
    assert result.exception is not None


def test_main_verbosity(
    mocker: pytest_mock.MockerFixture,
    sample_data_gt3x: pathlib.Path,
    create_typer_cli_runner: testing.CliRunner,
) -> None:
    """Test cli with different verbosity levels."""
    mock_run = mocker.patch.object(orchestrator, "run")

    create_typer_cli_runner.invoke(cli.app, [str(sample_data_gt3x), "-v"])

    mock_run.assert_called_once_with(
        input=sample_data_gt3x,
        output=None,
        thresholds=None,
        calibrator=None,
        activity_metric="enmo",
        epoch_length=5,
        nonwear_algorithm=["ggir"],
        verbosity=logging.DEBUG,
        output_filetype=".csv",
    )


def test_main_version(
    create_typer_cli_runner: testing.CliRunner,
) -> None:
    """Test cli version output."""
    result = create_typer_cli_runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert "Wristpy version" in result.output


def test_main_version_with_options(
    create_typer_cli_runner: testing.CliRunner,
    sample_data_gt3x: pathlib.Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test other arguments and options are ignored when --version is passed."""
    mock_run = mocker.patch.object(orchestrator, "run")

    result = create_typer_cli_runner.invoke(
        cli.app, [str(sample_data_gt3x), "-e", "5", "--version"]
    )

    assert result.exit_code == 0
    assert "Wristpy version" in result.output
    mock_run.assert_not_called()
