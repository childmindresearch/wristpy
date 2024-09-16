"""CLI argument parser for wristpy."""

import argparse
import pathlib
from typing import List, Optional

from wristpy.core import config

settings = config.Settings()


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Argument parser for python cli.

    Args:
        args: Optional argument for when accessed via python script. When default value
        (None) is given arg parser gets arguments from sys.argv.

    Returns:
        Namespace object with all of the input arguments.
    """
    parser = argparse.ArgumentParser(description="Run the main wristpy pipeline")

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
        choices=["ggir", "gradient"],
        default=None,
        help="Pick which calibrator to use. Can be 'ggir' or 'gradient."
        "None by default.",
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
        "In that order. Defaults to config values.",
    )

    return parser.parse_args(args)
