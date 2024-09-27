"""CLI for wristpy."""

import argparse
import pathlib
from typing import List, Optional

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
    default_settings = config.Settings()
    parser = argparse.ArgumentParser(
        description="Run the main wristpy pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Please report issues at https://github.com/childmindresearch/wristpy",
    )

    parser.add_argument("input", type=pathlib.Path, help="Path to the input data.")

    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="Path where data will be saved. Supports .csv and .parquet formats.",
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
            default_settings.LIGHT_THRESHOLD,
            default_settings.MODERATE_THRESHOLD,
            default_settings.VIGOROUS_THRESHOLD,
        ],
        help="Provide three thresholds for light, moderate, and vigorous activity."
        "Values must be given in ascending order.",
    )

    parser.add_argument(
        "-e",
        "--epoch_length",
        default=5,
        type=int,
        help="Specify the sampling rate in seconds for all metrics. Set to 0 to"
        "skip downsampling.",
    )

    parser.add_argument(
        "-v", "--version", action="version", version=config.get_version()
    )

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> orchestrator.Results:
    """Runs wristpy orchestrator with command line arguments.

    Args:
         args: A list of command line arguments given as strings. If None, the parser
            will take the args from `sys.argv`.

    Returns:
        A Results object containing enmo, anglez, physical activity levels, nonwear
        detection, and sleep detection.
    """
    arguments = parse_arguments(args)
    logger.debug("Running wristpy. arguments given: %s", arguments)

    light, moderate, vigorous = arguments.thresholds
    settings = config.Settings(
        LIGHT_THRESHOLD=light, MODERATE_THRESHOLD=moderate, VIGOROUS_THRESHOLD=vigorous
    )
    return orchestrator.run(
        input=arguments.input,
        output=arguments.output,
        settings=settings,
        calibrator=None if arguments.calibrator == "none" else arguments.calibrator,
        epoch_length=None if arguments.epoch_length == 0 else arguments.epoch_length,
    )
