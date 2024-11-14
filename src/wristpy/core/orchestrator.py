"""Python based runner."""

import datetime
import itertools
import logging
import pathlib
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import polars as pl

from wristpy.core import computations, config, exceptions, models
from wristpy.io.readers import readers
from wristpy.processing import analytics, calibration, metrics

logger = config.get_logger()

VALID_FILE_TYPES = (".csv", ".parquet")


def format_sleep_data(
    sleep_windows: List[analytics.SleepWindow], reference_measure: models.Measurement
) -> np.ndarray:
    """Formats sleep windows into an array for saving.

    Args:
        sleep_windows: The list of time stamp pairs indicating periods of sleep.
        reference_measure: The measure from which the temporal resolution will be taken.

    Returns:
        1-D np.ndarray, with 1 indicating sleep. Will be of the same length as
            the timestamps in the reference_measure.
    """
    logger.debug("Formatting sleep array from sleep windows.")
    sleep_array = np.zeros(len(reference_measure.time), dtype=bool)

    for window in sleep_windows:
        sleep_mask = (reference_measure.time >= window.onset) & (
            reference_measure.time <= window.wakeup
        )
        sleep_array[sleep_mask] = 1

    return sleep_array


def format_nonwear_data(
    nonwear_data: models.Measurement,
    reference_measure: models.Measurement,
    original_temporal_resolution: float,
) -> np.ndarray:
    """Formats nonwear data to match the temporal resolution of the other measures.

    The detect_nonweaer algorithm outputs non-wear values in 15-minute windows, where
    each timestamp represents the beginning of the window. This structure does not align
    well with the polars upsample function, which treats the last timestamp as the end
    of the time series. As a result, using the upsample function would cut off the
    final window. To avoid this, we manually map the non-wear data to the reference
    measure's resolution.

    Args:
        nonwear_data: The nonwear array to be upsampled.
        reference_measure: The measurement we match the non_wear data's temporal
            resolution to.
        original_temporal_resolution: The original temporal resolution of the
            nonwear_data.

    Returns:
        1-D np.ndarray with 1 indicating a nonwear timepoint. Will match the
            length of the reference measure.
    """
    logger.debug("Upsampling nonwear data.")
    nonwear_df = pl.DataFrame(
        {
            "nonwear": nonwear_data.measurements.astype(np.int64),
            "time": nonwear_data.time,
        }
    ).set_sorted("time")

    nonwear_array = np.zeros(len(reference_measure.time), dtype=bool)

    for row in nonwear_df.iter_rows(named=True):
        nonwear_value = row["nonwear"]
        time = row["time"]
        nonwear_mask = (reference_measure.time >= time) & (
            reference_measure.time
            <= time + datetime.timedelta(seconds=original_temporal_resolution)
        )
        nonwear_array[nonwear_mask] = nonwear_value

    return nonwear_array


def run(
    input: Union[pathlib.Path, str],
    output: Optional[Union[pathlib.Path, str]] = None,
    thresholds: Tuple[float, float, float] = (0.0563, 0.1916, 0.6958),
    calibrator: Union[
        None,
        Literal["ggir", "gradient"],
    ] = "gradient",
    epoch_length: Union[int, None] = 5,
    verbosity: int = logging.WARNING,
    output_dtype: Literal[".csv", ".parquet"] = ".csv",
) -> Optional[Union[models.Results, models.BatchedResults]]:
    """Runs main processing steps for wristpy as single files, or dirs."""
    logger.setLevel(verbosity)

    input = pathlib.Path(input)

    if input.is_file():
        logger.debug(f"Input is file, forwarding to run_file with output={output}")

        return run_file(
            input=input,
            output=output,
            thresholds=thresholds,
            calibrator=calibrator,
            epoch_length=epoch_length,
            verbosity=verbosity,
        )
    if output is not None:
        output = pathlib.Path(output)
        if not output.is_dir():
            raise ValueError(f"Output:{output} is not a directory.")

    file_names = [
        file for file in itertools.chain(input.glob("*.gt3x"), input.glob("*.bin"))
    ]

    if not file_names:
        raise ValueError(f"Directory {input} contains no .gt3x or .bin files.")

    batched_results = models.BatchedResults(results={})
    for file in file_names:
        output_file_path = (
            output / pathlib.Path(file.stem).with_suffix(output_dtype)
            if output
            else None
        )
        logger.debug(
            "Processing directory: %s, current file: %s, save path: %s",
            input,
            file,
            output_file_path,
        )
        try:
            result = run_file(
                input=input / file,
                output=output_file_path,
                thresholds=thresholds,
                calibrator=calibrator,
                epoch_length=epoch_length,
                verbosity=verbosity,
            )
            batched_results.add_result(file=file.stem, result=result)
        except Exception as e:
            logger.error("Did not run file: %s, Error: %s", file, e)

    return batched_results


def run_file(
    input: Union[pathlib.Path, str],
    output: Optional[Union[pathlib.Path, str]] = None,
    thresholds: Tuple[float, float, float] = (0.0563, 0.1916, 0.6958),
    calibrator: Union[
        None,
        Literal["ggir", "gradient"],
    ] = "gradient",
    epoch_length: Union[int, None] = 5,
    verbosity: int = logging.WARNING,
) -> models.Results:
    """Runs main processing steps for wristpy and returns data for analysis.

    The run() function will provide the user with enmo, anglez, physical activity levels
    sleep detection and nonwear data. All measures will be in the same temporal
    resolution. Users may choose from 'ggir' and 'gradient' calibration methods, or
    enter None to proceed without calibration.

    Args:
        input: Path to the input file to be read. Currently supports .bin and .gt3x
        output: Path to save data to. The path should end in the save file name in
            either .csv or .parquet formats.
        thresholds: The cut points for the light, moderate, and vigorous thresholds,
            given in that order. Values must be asscending, unique, and greater than 0.
            Default values are optimized for subjects ages 7-11 [1].
        calibrator: The calibrator to be used on the input data.
        epoch_length: The temporal resolution in seconds, the data will be down sampled
            to. If None is given no down sampling is preformed.
        verbosity: The logging level for the logger.

    Returns:
        All calculated data in a save ready format as a Results object.

    References:
        [1] Hildebrand, M., et al. (2014). Age group comparability of raw accelerometer
        output from wrist- and hip-worn monitors. Medicine and Science in Sports and
        Exercise, 46(9), 1816-1824.
    """
    logger.setLevel(verbosity)
    input = pathlib.Path(input)
    if output is not None:
        output = pathlib.Path(output)
        models.Results.validate_output(output=output)

    if calibrator is not None and calibrator not in ["ggir", "gradient"]:
        msg = (
            f"Invalid calibrator: {calibrator}. Choose: 'ggir', 'gradient'. "
            "Enter None if no calibration is desired."
        )
        logger.error(msg)
        raise ValueError(msg)

    watch_data = readers.read_watch_data(input)

    calibrator_object: Union[
        calibration.GgirCalibration, calibration.ConstrainedMinimizationCalibration
    ]
    if calibrator is None:
        logger.debug("Running without calibration")
        calibrated_acceleration = watch_data.acceleration
    else:
        if calibrator == "ggir":
            calibrator_object = calibration.GgirCalibration()
        elif calibrator == "gradient":
            calibrator_object = calibration.ConstrainedMinimizationCalibration()
        else:
            raise ValueError(f"Invalid calibrator given:{calibrator}")

        logger.debug("Running calibration with calibrator: %s", calibrator)
        try:
            calibrated_acceleration = calibrator_object.run_calibration(
                watch_data.acceleration
            )
        except (
            exceptions.CalibrationError,
            exceptions.SphereCriteriaError,
            exceptions.NoMotionError,
        ) as exc_info:
            logger.error(
                "Calibration FAILED: %s. Proceeding without calibration.", exc_info
            )
            calibrated_acceleration = watch_data.acceleration

    enmo = metrics.euclidean_norm_minus_one(calibrated_acceleration)
    anglez = metrics.angle_relative_to_horizontal(calibrated_acceleration)
    sleep_detector = analytics.GgirSleepDetection(anglez)
    sleep_windows = sleep_detector.run_sleep_detection()

    if epoch_length is not None:
        enmo = computations.moving_mean(enmo, epoch_length=epoch_length)
        anglez = computations.moving_mean(anglez, epoch_length=epoch_length)
    non_wear_array = metrics.detect_nonwear(calibrated_acceleration)
    physical_activity_levels = analytics.compute_physical_activty_categories(
        enmo,
        thresholds,
    )
    sleep_array = models.Measurement(
        measurements=format_sleep_data(
            sleep_windows=sleep_windows, reference_measure=enmo
        ),
        time=enmo.time,
    )
    nonwear_epoch = models.Measurement(
        measurements=format_nonwear_data(
            nonwear_data=non_wear_array,
            reference_measure=enmo,
            original_temporal_resolution=900,
        ),
        time=enmo.time,
    )

    results = models.Results(
        enmo=enmo,
        anglez=anglez,
        physical_activity_levels=physical_activity_levels,
        sleep_windows_epoch=sleep_array,
        nonwear_epoch=nonwear_epoch,
    )
    if output is not None:
        try:
            results.save_results(output=output)
        except (
            exceptions.InvalidFileTypeError,
            exceptions.DirectoryNotFoundError,
        ) as exc_info:
            # Allowed to pass to recover in Jupyter Notebook scenarios.
            logger.error(
                (
                    f"Could not save output due to: {exc_info}. Call save_results "
                    " on the output object with a correct filename to save these "
                    "results."
                )
            )

    return results
