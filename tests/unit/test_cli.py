"""Test the wristpy cli."""

import logging
import pathlib

import pytest
import pytest_mock
import typer
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

    result = create_typer_cli_runner.invoke(cli.app, [str(sample_data_gt3x)])

    assert result.exit_code == 0
    mock_run.assert_called_once_with(
        input=sample_data_gt3x,
        output=None,
        thresholds=None,
        calibrator=None,
        activity_metric=["enmo"],
        epoch_length=5,
        nonwear_algorithm=["ggir"],
        verbosity=logging.INFO,
        output_filetype=".csv",
    )


@pytest.mark.parametrize(
    "metric",
    [
        "enmo",
        "mad",
        "ag_count",
        "mims",
    ],
)
def test_main_with_metrics(
    mocker: pytest_mock.MockerFixture,
    sample_data_gt3x: pathlib.Path,
    create_typer_cli_runner: testing.CliRunner,
    metric: str,
) -> None:
    """Test that correct ag_count default thresholds are pulled."""
    mock_run = mocker.patch.object(orchestrator, "_run_file")

    create_typer_cli_runner.invoke(cli.app, ([str(sample_data_gt3x), "-a", metric]))

    mock_run.assert_called_once_with(
        input=sample_data_gt3x,
        output=None,
        thresholds=None,
        calibrator=None,
        activity_metric=[metric],
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
                "0.1 1.0 1.5",
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
        thresholds=[(0.1, 1.0, 1.5)],
        calibrator="gradient",
        activity_metric=["mad"],
        nonwear_algorithm=["cta", "ggir"],
        epoch_length=3,
        verbosity=logging.INFO,
        output_filetype=".csv",
    )


def test_main_with_wrong_number_of_thresholds(
    sample_data_gt3x: pathlib.Path,
    create_typer_cli_runner: testing.CliRunner,
) -> None:
    """Test cli with bad thresholds."""
    result = create_typer_cli_runner.invoke(
        cli.app, [str(sample_data_gt3x), "-a", "enmo", "-a", "mad", "-t", "1 2 3"]
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "Number of thresholds did not match the number of activity metrics." in str(
        result.exception
    )


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
        activity_metric=["enmo"],
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


@pytest.mark.parametrize(
    "thresholds_str_lst, threshold_flt_lst",
    [
        (["0.1 0.5 1.0"], [(0.1, 0.5, 1.0)]),
        (["0.2 0.4 0.6", "0.3 0.5 0.7"], [(0.2, 0.4, 0.6), (0.3, 0.5, 0.7)]),
    ],
)
def test_parse_thresholds_valid(
    thresholds_str_lst: list[str], threshold_flt_lst: list[float]
) -> None:
    """Test that valid thresholds are parsed correctly."""
    parsed = cli._parse_thresholds(thresholds_str_lst)

    assert parsed == threshold_flt_lst


def test_parse_thresholds_invalid_triplet() -> None:
    """Tests that giving less than 3 numbers will error."""
    thresholds = ["1"]
    with pytest.raises(
        typer.BadParameter, match="Threshold triplet must have exactly 3 floats: 1"
    ):
        cli._parse_thresholds(thresholds)


def test_parse_thresholds_invalid_flt() -> None:
    """Tests that giving a non float will error."""
    thresholds = ["1 2 not_a_float"]
    with pytest.raises(
        typer.BadParameter, match="Invalid float in threshold: 1 2 not_a_float"
    ):
        cli._parse_thresholds(thresholds)
