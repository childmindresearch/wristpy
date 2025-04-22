"""CLI for wristpy."""

import logging
import pathlib
from typing import List, Literal, Optional, Tuple, Union, cast

import typer

from wristpy.core import config, orchestrator

logger = config.get_logger()
app = typer.Typer(
    help="Run the main Wristpy pipeline.",
    epilog="Please report issues at https://github.com/childmindresearch/wristpy.",
)


def _none_or_float_list(value: str) -> Optional[List[float]]:
    """Helper function to process thresholds argument."""
    if value.lower() == "none":
        return None
    try:
        float_list = [float(v) for v in value.split(",")]
        if len(float_list) != 3:
            raise typer.BadParameter(
                f"Invalid value: {value}."
                "Must be a comma-separated list of exactly three numbers or 'None'."
            )
        return float_list
    except ValueError:
        raise typer.BadParameter(
            f"Invalid value: {value}. Must be a comma-separated list or 'None'."
        )


def _parse_nonwear_algorithms(algorithm_name: str) -> List[str]:
    """Parse comma-separated non-wear algorithm names."""
    valid_algorithm_names = ["ggir", "cta", "detach"]
    algorithms = [algo.strip().lower() for algo in algorithm_name.split(",")]
    for algo in algorithms:
        if algo not in valid_algorithm_names:
            raise typer.BadParameter(
                f"Invalid algorithm: '{algo}'. Must be one of: "
                f"{', '.join(valid_algorithm_names)}."
            )
    return algorithms


@app.command()
def main(
    input: pathlib.Path = typer.Argument(
        ..., help="Path to the input data.", exists=True
    ),
    output: Optional[pathlib.Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Path where data will be saved. Supports .csv and .parquet formats.",
    ),
    output_filetype: Optional[str] = typer.Option(
        None,
        "-O",
        "--output-filetype",
        help="Format for save files when processing directories. "
        "Leave as None when processing single files.",
    ),
    calibrator: Union[
        None,
        Literal["ggir", "gradient"],
    ] = typer.Option(
        "none",
        "-c",
        "--calibrator",
        help="Pick which calibrator to use.",
        case_sensitive=False,
        callback=lambda x: x.lower() if x else x,
    ),
    activity_metric: str = typer.Option(
        "enmo",
        "-a",
        "--activity-metric",
        help="Pick which physical activity metric should be used for physical activity categorization.",
        case_sensitive=False,
        callback=lambda x: x.lower() if x else x,
    ),
    thresholds: Optional[str] = typer.Option(
        None,
        "-t",
        "--thresholds",
        help="Provide three thresholds for light, moderate, and vigorous activity. "
        "Exactly three values must be given in ascending order, and comma seperated.",
        callback=_none_or_float_list,
    ),
    nonwear_algorithm: List[str] = typer.Option(
        ["ggir"],
        "-nw",
        "--nonwear-algorithm",
        help="Specify the non-wear detection algorithm(s) to use. "
        "Specify one or more of 'ggir', 'cta', 'detach' as a comma-separated list "
        "(e.g. 'ggir,detach'). "
        "When multiple algorithms are specified, majority voting will be applied.",
        callback=_parse_nonwear_algorithms,
    ),
    epoch_length: int = typer.Option(
        5,
        "-e",
        "--epoch-length",
        help="Specify the sampling rate in seconds for all metrics. "
        "Must be greater than 0.",
        min=0,
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

    logger.debug("Running wristpy. arguments given: %s", locals())
    orchestrator.run(
        input=input,
        output=output,
        calibrator=None if calibrator == "none" else calibrator,
        activity_metric=activity_metric,
        thresholds=None
        if thresholds is None
        else cast(Tuple[float, float, float], tuple(thresholds)),
        epoch_length=epoch_length,
        nonwear_algorithm=nonwear_algorithm,
        verbosity=log_level,
        output_filetype=output_filetype,
    )


if __name__ == "__main__":
    app()
