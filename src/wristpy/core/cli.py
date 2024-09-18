"""CLI for wristpy."""

import argparse
import pathlib
from typing import List, Optional

from wristpy.core import config, orchestrator

logger = config.get_logger()
settings = config.Settings()


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Argument parser for python cli.

    Args:
        args: Optional argument for when accessed via python script.

    Returns:
        Namespace object with all of the input arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the main wristpy pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input", type=pathlib.Path, help="Path to the input data.")

    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="Optional path where data will be saved.",
    )

    parser.add_argument(
        "-c",
        "--calibrator",
        type=str,
        choices=["ggir", "gradient", "none"],
        help="Pick which calibrator to use. Can be 'ggir' or 'gradient'."
        "Leave empty or enter 'none' to proceed without calibration.",
    )

    parser.add_argument(
        "-t",
        "--thresholds",
        type=float,
        nargs=3,
        default=[
            settings.LIGHT_THRESHOLD,
            settings.MODERATE_THRESHOLD,
            settings.VIGOROUS_THRESHOLD,
        ],
        help="Provide three thresholds for light, moderate, and vigorous activity."
        "values must be given in ascending order. Defaults to config values.",
    )

    parser.add_argument(
        "-e",
        "--epoch_length",
        type=int,
        help="Desired sampling rate in seconds. Leave empty to skip downsampling.",
    )

    parser.add_argument(
        "-v", "--version", action="version", version=config.get_version()
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    arguments = parse_arguments()
    logger.debug("Running wristpy. arguments given: %s", arguments)
    light, moderate, vigorous = arguments.thresholds
    settings = config.Settings(
        LIGHT_THRESHOLD=light, MODERATE_THRESHOLD=moderate, VIGOROUS_THRESHOLD=vigorous
    )
    orchestrator.run(
        input=arguments.input,
        output=arguments.output,
        settings=settings,
        calibrator=arguments.calibrator,
        epoch_length=arguments.epoch_length,
    )
