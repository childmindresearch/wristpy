"""CLI for wristpy."""

import logging
import pathlib
from enum import Enum
from typing import List, Literal, Optional, Tuple, Union, cast

import typer

from wristpy.core import config

logger = config.get_logger()
app = typer.Typer(
    help="Run the main Wristpy pipeline.",
    epilog="Please report issues at https://github.com/childmindresearch/wristpy.",
)


class OutputFileType(str, Enum):
    """Valid output file types for saving data."""

    csv = "csv"
    parquet = "parquet"


class Calibrator(str, Enum):
    """Setting a calibrator class for typer.

    This class is used to define the literal types that are allowed for
    calibration, and parsing the strings for the orchestrator.
    """

    none = "none"
    ggir = "ggir"
    gradient = "gradient"


class ActivityMetric(str, Enum):
    """Valid activity metrics for physical activity categorization."""

    enmo = "enmo"
    mad = "mad"
    ag_count = "ag_count"


class NonwearAlgorithms(str, Enum):
    """Setting a nonwear algorithm class for typer.

    This class is used to define the literal types that are allowed for
    nonwear algorithms, and parsing the strings for the orchestrator.
    """

    ggir = "ggir"
    cta = "cta"
    detach = "detach"


@app.command()
def main(
    input: pathlib.Path = typer.Argument(
        ..., help="Path to the input data.", exists=True
    ),
    output: pathlib.Path = typer.Option(
        None,
        "-o",
        "--output",
        help="Path where data will be saved. Supports .csv and .parquet formats.",
    ),
    output_filetype: OutputFileType = typer.Option(
        None,
        "-O",
        "--output-filetype",
        help="Format for save files when processing directories. "
        "Leave as None when processing single files.",
    ),
    calibrator: Calibrator = typer.Option(
        None,
        "-c",
        "--calibrator",
        help="Pick which calibrator to use."
        "Must choose one of 'none', 'ggir', or 'gradient'.",
        case_sensitive=False,
    ),
    activity_metric: ActivityMetric = typer.Option(
        ActivityMetric.enmo,
        "-a",
        "--activity-metric",
        help="Metric should be used for physical activity categorization."
        "Choose from 'enmo', 'mad', or 'ag_count'.",
        case_sensitive=False,
    ),
    thresholds: tuple[float, float, float] = typer.Option(
        None,
        "-t",
        "--thresholds",
        help="Provide three thresholds for light, moderate, and vigorous activity. "
        "Exactly three values must be >= 0, given in ascending order,"
        " and separated by a space.",
        min=0,
    ),
    nonwear_algorithm: list[NonwearAlgorithms] = typer.Option(
        [NonwearAlgorithms.ggir],
        "-n",
        "--nonwear-algorithm",
        help="Specify the non-wear detection algorithm(s) to use. "
        "Specify one or more of 'ggir', 'cta', 'detach'. "
        "(e.g. '-n ggir -n cta'). "
        "When multiple algorithms are specified, majority voting will be applied.",
    ),
    epoch_length: int = typer.Option(
        5,
        "-e",
        "--epoch-length",
        help="Specify the sampling rate in seconds for all metrics. "
        "Must be greater than 1.",
        min=1,
    ),
    verbosity: int = typer.Option(
        0,
        "-v",
        "--verbosity",
        count=True,
        help="Determines the level of verbosity. Use -v for info, -vv for debug. "
        "Default for warning.",
    ),
    version: bool = typer.Option(
        False, "-V", "--version", help="Show the version and exit."
    ),
) -> None:
    """Run wristpy orchestrator with command line arguments."""
    from wristpy.core import orchestrator

    if version:
        typer.echo(f"Wristpy version: {config.get_version()}")
        raise typer.Exit()

    if verbosity == 0:
        log_level = logging.WARNING
    elif verbosity == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    logger.setLevel(log_level)

    nonwear_algorithms = [algo.value for algo in nonwear_algorithm]

    logger.debug("Running wristpy. arguments given: %s", locals())
    orchestrator.run(
        input=input,
        output=output,
        calibrator=calibrator.value if calibrator else None,  # type: ignore[arg-type] # Covered by Calibrator Enum class
        activity_metric=activity_metric.value,
        thresholds=None if thresholds is None else thresholds,
        epoch_length=epoch_length,
        nonwear_algorithm=nonwear_algorithms,  # type: ignore[arg-type] # Covered by NonwearAlgorithm Enum class
        verbosity=log_level,
        output_filetype=output_filetype,  # type: ignore[arg-type] # Covered by OutputFileType Enum class
    )


if __name__ == "__main__":
    app()
