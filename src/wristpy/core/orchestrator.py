"""Python based runner."""

import datetime
import pathlib
from typing import List, Literal, Optional, Union

import numpy as np
import polars as pl
import pydantic

from wristpy.core import computations, config, exceptions, models
from wristpy.io.readers import readers
from wristpy.processing import analytics, calibration, metrics

logger = config.get_logger()

VALID_FILE_TYPES = (".csv", ".parquet")


class Results(pydantic.BaseModel):
    """dataclass containing results of orchestrator.run()."""

    enmo: models.Measurement
    anglez: models.Measurement
    physical_activity_levels: models.Measurement
    nonwear_epoch: models.Measurement
    sleep_windows_epoch: models.Measurement

    def save_results(self, output: pathlib.Path) -> None:
        """Convert to polars and save the dataframe as a csv or parquet file.

        Args:
            output: The path and file name of the data to be saved. as either a csv or
                parquet files.

        """
        logger.debug("Saving results.")
        self.validate_output(output=output)

        results_dataframe = pl.DataFrame(
            {"time": self.enmo.time}
            | {name: value.measurements for name, value in self}
        )

        if output.suffix == ".csv":
            results_dataframe.write_csv(output, separator=",")
        elif output.suffix == ".parquet":
            results_dataframe.write_parquet(output)
        else:
            raise exceptions.InvalidFileTypeError(
                f"File type must be one of {VALID_FILE_TYPES}"
            )

        logger.debug("results saved in: %s", output)

    @classmethod
    def validate_output(cls, output: pathlib.Path) -> None:
        """Validates that the output path exists and is a valid format.

        Args:
            output: the name of the file to be saved, and the directory it will
            be saved in. Must be a .csv or .parquet file.

        Raises:
            InvalidFileTypeError:If the output file path ends with any extension other
                    than csv or parquet.
            DirectoryNotFoundError: If the directory for the file to be saved in does
                not exist.
        """
        if not output.parent.exists():
            raise exceptions.DirectoryNotFoundError(
                f"The directory:{output.parent} does not exist."
            )
        if output.suffix not in VALID_FILE_TYPES:
            raise exceptions.InvalidFileTypeError(
                f"The extension: {output.suffix} is not supported."
                "Please save the file as .csv or .parquet",
            )


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
    settings: config.Settings = config.Settings(),
    calibrator: Union[
        None,
        Literal["ggir", "gradient"],
    ] = "gradient",
    epoch_length: Union[int, None] = 5,
) -> Results:
    """Runs main processing steps for wristpy and returns data for analysis.

    The run() function will provide the user with enmo, anglez, physical activity levels
    sleep detection and nonwear data. All measures will be in the same temporal
    resolution. Users may choose from 'ggir' and 'gradient' calibration methods, or
    enter None to proceed without calibration.

    Args:
        input: Path to the input file to be read. Currently supports .bin and .gt3x
        output: Path to save data to. The path should end in the save file name in
            either .csv or .parquet formats.
        settings: The settings object from which physical activity levels are taken.
        calibrator: The calibrator to be used on the input data.
        epoch_length: The temporal resolution in seconds, the data will be down sampled
            to. If None is given no down sampling is preformed.

    Returns:
        All calculated data in a save ready format as a Results object.

    """
    input = pathlib.Path(input)
    if output is not None:
        output = pathlib.Path(output)
        Results.validate_output(output=output)

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
        else:
            calibrator_object = calibration.ConstrainedMinimizationCalibration()

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
    if epoch_length is not None:
        enmo = computations.moving_mean(enmo, epoch_length=epoch_length)
        anglez = computations.moving_mean(anglez, epoch_length=epoch_length)
    if settings.RANGE_CRITERIA is not None:
        non_wear_array = metrics.detect_nonwear(
            calibrated_acceleration, range_criteria=settings.RANGE_CRITERIA
        )
    else:
        non_wear_array = metrics.detect_nonwear(calibrated_acceleration)

    sleep_detector = analytics.GGIRSleepDetection(anglez)
    sleep_windows = sleep_detector.run_sleep_detection()
    physical_activity_levels = analytics.compute_physical_activty_categories(
        enmo,
        (
            settings.LIGHT_THRESHOLD,
            settings.MODERATE_THRESHOLD,
            settings.VIGOROUS_THRESHOLD,
        ),
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

    results = Results(
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
