"""Python based runner."""

import datetime
import pathlib
from typing import List, Literal, Optional, Union

import numpy as np
import polars as pl
import pydantic

from wristpy.core import computations, config, models
from wristpy.io.readers import readers
from wristpy.processing import analytics, calibration, metrics

logger = config.get_logger()


class InvalidFileTypeError(calibration.LoggedException):
    """Wristpy cannot save in the given file type."""

    pass


class DirectoryNotFoundError(calibration.LoggedException):
    """Output save path not found."""

    pass


class Results(pydantic.BaseModel):
    """dataclass containing results of orchestrator.run()."""

    enmo: Optional[models.Measurement] = None
    anglez: Optional[models.Measurement] = None
    physical_activity_levels: Optional[models.Measurement] = None
    nonwear_epoch1: Optional[models.Measurement] = None
    sleep_windows_epoch1: Optional[models.Measurement] = None

    def save_results(self, output: pathlib.Path) -> None:
        """Convert to polars and save the dataframe as a csv or parquet file.

        The data captured by Results can be in one of two temporal resolutions, the
        unaltered (raw) time stamps taken from the watch data, and the down sampled
        epoch1 time stamps (5 second intervals). The "raw" time is used in the enmo and
        anglez data, and the epoch1 time is used for enmo_epoch1, anglez_epoch1,
        nonwear_epoch1, physical_activity_levels and sleep_windows_epoch1. nonwear_array
        values are in 15 minute blocks and are upsampled to match epoch1 time for the
        purposes of saving. Additionally the sleep window data is a list of timestamp
        pairs, which are used to create a binary array in epoch1 time. Two files will
        be saved, one of each temporal resolution. This means if output is entered as
        /path/to/file/file.csv,The file names will be labeled as file_epoch1.csv and
        file_raw_time.csv in the path/to/file directory.

        Args:
            output: The path and file name of the data to be saved. as either a csv or
                parquet files.

        """
        validate_output(output=output)

        results_dataframe = pl.DataFrame({"time": self.enmo.time})

        for field_name, field_value in self:
            results_dataframe = results_dataframe.with_columns(
                pl.Series(field_name, field_value.measurements)
            )

        if output.suffix == ".csv":
            results_dataframe.write_csv(output, separator=",")
        elif output.suffix == ".parquet":
            results_dataframe.write_parquet(output)


def validate_output(output: pathlib.Path) -> None:
    """Validates that the output path exists and is a valid format.

    Args:
        output: the name of the file to be saved, and the directory it will be saved in.
            must be a .csv or .parquet file.

    Raises:
        InvalidFileTypeError:If the output file path ends with any extension other
                than csv or parquet.
        DirectoryNotFoundError: If the directory for the file to be saved in does not
            exist.
    """
    if not output.parent.exists():
        raise DirectoryNotFoundError(f"The directory:{output.parent} does not exist.")
    if output.suffix not in [".csv", ".parquet"]:
        raise InvalidFileTypeError(
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
    sleep_array = np.zeros(len(reference_measure.time))

    for window in sleep_windows:
        sleep_mask = (reference_measure.time >= window.onset) & (
            reference_measure.time <= window.wakeup
        )
        sleep_array[sleep_mask] = 1

    return sleep_array


def format_nonwear_data(
    nonwear_data: models.Measurement,
    reference_measure: models.Measurement,
    nonwear_temporal_resolution: float,
) -> np.ndarray:
    """Here."""
    nonwear_df = pl.DataFrame(
        {
            "nonwear": nonwear_data.measurements.astype(np.int64),
            "time": nonwear_data.time,
        }
    ).set_sorted("time")

    nonwear_array = np.zeros(len(reference_measure.time))

    for row in nonwear_df.iter_rows(named=True):
        nonwear_value = row["nonwear"]
        time = row["time"]
        nonwear_mask = (reference_measure.time >= time) & (
            reference_measure.time
            <= time + datetime.timedelta(seconds=nonwear_temporal_resolution)
        )
        nonwear_array[nonwear_mask] = nonwear_value

    return nonwear_array


def run(
    input: pathlib.Path,
    output: Optional[pathlib.Path] = None,
    settings: config.Settings = config.Settings(LOGGING_LEVEL=10),
    calibrator: Union[
        None,
        Literal["ggir", "gradient"],
    ] = "gradient",
    epoch_length: Union[int, None] = 5,
) -> Results:
    """Runs wristpy.

    Args:
        input: Path to the input file to be read. Currently supports .bin and .gt3x
        output: Path to save data to. The path should end in the file name to be given
            to the save data. Two files will be saved, each with the given file name and
            the _raw_time or _epoch1 label after. Currently supports saving in .csv and
            .parquet
        settings: The settings object from which physical activity levels are taken.
        calibrator: The calibrator to be used on the input data.
        epoch_length: The temporal resolution in seconds, the data will be down sampled to.

    Returns:
        All calculated data in a save ready format as a Results object.

    """
    if output is not None:
        validate_output(output=output)

    if calibrator is not None and calibrator not in ["ggir", "gradient"]:
        raise ValueError(
            f"Invalid calibrator: {calibrator}. Choose: 'ggir', 'gradient'. "
            "Enter None if no calibration is desired.",
        )

    watch_data = readers.read_watch_data(input)
    if calibrator == "ggir":
        calibrator = calibration.GgirCalibration()
    elif calibrator == "gradient":
        calibrator = calibration.ConstrainedMinimizationCalibration()

    if calibrator is None:
        logger.debug("Running without calibration.")
        calibrated_acceleration = watch_data.acceleration
    else:
        try:
            logger.debug("Running calibration with calibrator: %s", calibrator)
            calibrated_acceleration = calibrator.run_calibration(
                watch_data.acceleration
            )
        except (
            ValueError,
            calibration.CalibrationError,
            calibration.ZeroScaleError,
            calibration.SphereCriteriaError,
            calibration.NoMotionError,
        ) as e:
            print(f"Calibration FAILED:{e}")
            print("Proceeding without calibration.")
            calibrated_acceleration = watch_data.acceleration

    enmo = metrics.euclidean_norm_minus_one(calibrated_acceleration)
    anglez = metrics.angle_relative_to_horizontal(calibrated_acceleration)
    if epoch_length is not None:
        enmo = computations.moving_mean(enmo, epoch_length=epoch_length)
        anglez = computations.moving_mean(anglez, epoch_length=epoch_length)
    if input.suffix == ".bin":
        non_wear_array = metrics.detect_nonwear(
            calibrated_acceleration, range_criteria=0.5
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
    nonwear_epoch1 = models.Measurement(
        measurements=format_nonwear_data(
            nonwear_data=non_wear_array,
            reference_measure=enmo,
            nonwear_temporal_resolution=900,
        ),
        time=enmo.time,
    )

    results = Results(
        enmo=enmo,
        anglez=anglez,
        physical_activity_levels=physical_activity_levels,
        sleep_windows_epoch1=sleep_array,
        nonwear_epoch1=nonwear_epoch1,
    )
    if output is not None:
        try:
            results.save_results(output=output)
        except (InvalidFileTypeError, DirectoryNotFoundError) as e:
            print(e)

    return results
