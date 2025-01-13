"""CLI for wristpy."""

import argparse
import logging
import pathlib
from typing import List, Optional, Tuple, cast

from wristpy.core import config, orchestrator

logger = config.get_logger()


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Argument parser for Wristpy's cli.

    Args:
        args: A list of command line arguments given as strings. If None, the parser
            will take the args from `sys.argv`.

    Returns:
        Namespace object with all of the input arguments and default values.
    """
    parser = argparse.ArgumentParser(
        description="Run the main Wristpy pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Please report issues at https://github.com/childmindresearch/wristpy.",
    )

    parser.add_argument("input", type=pathlib.Path, help="Path to the input data.")

    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="Path where data will be saved. Supports .csv and .parquet formats.",
    )

    parser.add_argument(
        "-O",
        "--output_filetype",
        type=str,
        default=None,
        help="Format for save files when processing directories. "
        "Leave as None when processing single files.",
    )

    parser.add_argument(
        "-c",
        "--calibrator",
        type=str,
        choices=["ggir", "gradient", "none"],
        default="none",
        help="Pick which calibrator to use. Can be 'ggir' or 'gradient'.",
    )

    parser.add_argument(
        "-a",
        "--activity_metric",
        type=str,
        choices=["ENMO", "MAD"],
        default="ENMO",
        help="Pick which physical activity metric should be used. "
        "This will be used to determine physical activity categorization. "
        "Can be 'ENMO' or 'MAD'.",
    )

    parser.add_argument(
        "-t",
        "--thresholds",
        type=_none_or_float_list,
        default=None,
        help="Provide three thresholds for light, moderate, and vigorous activity. "
        "Values must be given in ascending order, and comma seperated.",
    )

    parser.add_argument(
        "-e",
        "--epoch_length",
        default=5,
        type=int,
        help="Specify the sampling rate in seconds for all metrics. To skip resampling,"
        " enter 0.",
    )

    parser.add_argument(
        "-V", "--version", action="version", version=config.get_version()
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="Determines the level of verbosity. Use -v for info, -vv for debug."
        "Default for warning.",
    )

    return parser.parse_args(args)


def main(
    args: Optional[List[str]] = None,
) -> None:
    """Runs wristpy orchestrator with command line arguments.

    Args:
         args: A list of command line arguments given as strings. If None, the parser
            will take the args from `sys.argv`.

    Returns:
        A Results object containing enmo, anglez, physical activity levels, nonwear
        detection, and sleep detection.

    Raises:
        ValueError: If the epoch_length is less than 0.
    """
    arguments = parse_arguments(args)

    if arguments.verbosity == 0:
        log_level = logging.WARNING
    elif arguments.verbosity == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    logger.setLevel(log_level)

    if arguments.epoch_length < 0:
        raise ValueError(
            f"Value for epoch_length is:{arguments.epoch_length}."
            "Please enter an integer >= 0."
        )

    logger.debug("Running wristpy. arguments given: %s", arguments)

    orchestrator.run(
        input=arguments.input,
        output=arguments.output,
        calibrator=None if arguments.calibrator == "none" else arguments.calibrator,
        activity_metric=arguments.activity_metric,
        thresholds=None
        if arguments.thresholds is None
        else cast(Tuple[float, float, float], tuple(arguments.thresholds)),
        epoch_length=None if arguments.epoch_length == 0 else arguments.epoch_length,
        verbosity=log_level,
        output_filetype=arguments.output_filetype,
    )


def _none_or_float_list(value: str) -> Optional[List[float]]:
    """Helper function to process thresholds argument."""
    if value.lower() == "none":
        return None
    try:
        return [float(v) for v in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid value: {value}. Must be a comma-separated list or 'None'."
        )
