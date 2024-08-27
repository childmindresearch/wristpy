"""CLI argument parser for wristpy."""

import argparse
import pathlib
from typing import List, Optional

from wristpy.core import config
from wristpy.processing import calibration


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Argument parser for python cli.

    Args:
        args: Optional argument for when accessed via python script. When default value
        (None) is given arg parser gets arguments from sys.argv.

    Returns:
        Namespace object will all of the input arguments. s
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
        "-l",
        "--light-threshold",
        type=float,
        default=config.Settings().LIGHT_THRESHOLD,
        help="Threshold for light physical activity",
    )
    parser.add_argument(
        "-m",
        "--moderate-threshold",
        type=float,
        default=config.Settings().MODERATE_THRESHOLD,
        help="Threshold for moderate physical activity",
    )
    parser.add_argument(
        "-v",
        "--vigorous-threshold",
        type=float,
        default=config.Settings().VIGOROUS_THRESHOLD,
        help="Threshold for vigorous physical activity",
    )

    parser.add_argument(
        "--chunked", action="store_false", help="Use chunked calibration"
    )
    parser.add_argument(
        "--min-acceleration",
        type=float,
        default=calibration.Calibration().min_acceleration,
        help="Minimum acceleration for sphere criteria",
    )
    parser.add_argument(
        "--min-hours",
        type=int,
        default=calibration.Calibration().min_calibration_hours,
        help="Minimum hours of data required for calibration",
    )
    parser.add_argument(
        "--min-std",
        type=float,
        default=calibration.Calibration().min_standard_deviation,
        help="Minimum standard deviation for no-motion detection",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=calibration.Calibration().max_iterations,
        help="Maximum number of iterations for optimization",
    )
    parser.add_argument(
        "--error-tolerance",
        type=float,
        default=calibration.Calibration().error_tolerance,
        help="Tolerance for optimization convergence",
    )
    parser.add_argument(
        "--min-calibration-error",
        type=float,
        default=calibration.Calibration().min_calibration_error,
        help="Threshold for calibration error",
    )

    parser.add_argument(
        "--short-length",
        type=int,
        default=900,
        help="Short window size for non-wear detection, in seconds",
    )
    parser.add_argument(
        "--short-in-long",
        type=int,
        default=4,
        help="Number of short epochs that make up one long epoch",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=0.013,
        help="Threshold criteria for standard deviation in non-wear detection",
    )
    parser.add_argument(
        "--range",
        type=float,
        default=0.05,
        help="Threshold criteria for range of acceleration in non-wear detection",
    )

    return parser.parse_args(args)
