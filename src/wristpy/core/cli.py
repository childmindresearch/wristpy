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
        "-t",
        "--thresholds",
        type=float,
        nargs=3,
        default=[
            0.0563,
            0.1916,
            0.6958,
        ],
        help="Provide three thresholds for light, moderate, and vigorous activity. "
        "Values must be given in ascending order.",
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

    if not (
        0 < arguments.thresholds[0] < arguments.thresholds[1] < arguments.thresholds[2]
    ):
        message = "Threshold values must be >=0, unique, and in ascending order."
        logger.error(message)
        raise ValueError(message)

    logger.debug("Running wristpy. arguments given: %s", arguments)

    orchestrator.run(
        input=arguments.input,
        output=arguments.output,
        thresholds=cast(Tuple[float, float, float], tuple(arguments.thresholds)),
        calibrator=None if arguments.calibrator == "none" else arguments.calibrator,
        epoch_length=None if arguments.epoch_length == 0 else arguments.epoch_length,
        verbosity=log_level,
        output_filetype=arguments.output_filetype,
    )
