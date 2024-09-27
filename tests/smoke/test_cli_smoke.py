"""Smoke tests for wristpy cli."""

import pathlib

import pytest_mock

from wristpy.core import cli, config, orchestrator


def test_main_default(
    mocker: pytest_mock.MockerFixture, sample_data_gt3x: pathlib.Path
) -> None:
    """Test Cli with only necessary arguments."""
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
    """Test Cli with only necessary arguments."""
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
