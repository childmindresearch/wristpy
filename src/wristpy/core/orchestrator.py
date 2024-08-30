"""Python based runner."""

import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from pydantic import BaseModel

from wristpy.core import computations, config, models
from wristpy.io.readers import readers
from wristpy.processing import analytics, calibration, metrics


class InvalidFileTypeError(Exception):
    """Wristpy cannot save in the given file type."""

    pass


class Results(BaseModel):
    """dataclass containing results of orchestrator.run()."""

    enmo: Optional[models.Measurement] = None
    anglez: Optional[models.Measurement] = None
    enmo_epoch1: Optional[models.Measurement] = None
    anglez_epoch1: Optional[models.Measurement] = None
    nonwear_array: Optional[models.Measurement] = None
    sleep_windows: Optional[List[analytics.SleepWindow]] = None
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

        Returns:
            None.

        Raises:
            InvalidFileTypeError: If the output file path ends with any extension other
                than csv or parquet.
        """
        if output.suffix not in [".csv", ".parquet"]:
            raise InvalidFileTypeError(
                f"The extension: {output.suffix} is not supported.",
                "Please save the file as .csv or .parquet",
            )

        results_epoch1_time = self._results_to_dataframe()
        results_raw_time = self._results_to_dataframe(use_epoch1_time=False)

        save_path_epoch1 = output.with_name(output.stem + "_epoch1" + output.suffix)
        save_path_raw_time = output.with_name(output.stem + "_raw_time" + output.suffix)

        if output.suffix == ".csv":
            results_epoch1_time.write_csv(save_path_epoch1, separator=",")
            results_raw_time.write_csv(save_path_raw_time, separator=",")
        elif output.suffix == ".parquet":
            results_epoch1_time.write_parquet(save_path_epoch1)
            results_raw_time.write_parquet(save_path_raw_time)

    def _results_to_dataframe(self, use_epoch1_time: bool = True) -> pl.DataFrame:
        """Format results into a dataframe.

        Args:
            use_epoch1_time: Determines which temporal resolution and data to use. If
                True the dataframe will contain enmo_epoch1, anglez_epoch1,
                nonwear_epoch1, physical_activity_levels and sleep_windows_epoch1. If
                False the the dataframe will be in "raw" time and contain the enmo and
                anglez data.

        Returns:
            The polars dataframe to be saved into csv or parquet format.

        Raises:
            ValueError if the attribute for the time column is None.
        """
        data_source = self.enmo_epoch1 if use_epoch1_time else self.enmo

        if data_source is None:
            raise ValueError(f"{data_source} is None, can't construct time column.")

        time_data = data_source.time

        results_dataframe = pl.DataFrame({"time": time_data})

        for field_name, field_value in self:
            if field_value is not None:
                if (
                    use_epoch1_time
                    and field_name
                    not in ["enmo", "anglez", "sleep_windows", "nonwear_array"]
                ) or (not use_epoch1_time and field_name in ["enmo", "anglez"]):
                    results_dataframe = results_dataframe.with_columns(
                        pl.Series(field_name, field_value.measurements)
                    )

        return results_dataframe


def format_sleep_data(
    sleep_windows: List[analytics.SleepWindow], reference_measure: models.Measurement
) -> np.ndarray:
    """Formats sleep windows into an array for saving.

    Args:
        sleep_windows: The list of time stamp pairs indicating periods of sleep.
        reference_measure: The measure from which the temporal resolution will be taken.
            Any epoch1 measure can be used, but enmo is used for consistency.

    Returns:
        1-D binary np.ndarray, with 1 indicating sleep. Will be of the same length as
            the timestamps in the epoch1_measure.
    """
    sleep_array = np.zeros(len(reference_measure.time))

    for window in sleep_windows:
        sleep_mask = (reference_measure.time >= window.onset) & (
            reference_measure.time <= window.wakeup
        )
        sleep_array[sleep_mask] = 1

    return sleep_array


def format_nonwear_data(
    nonwear_data: models.Measurement, reference_measure: models.Measurement
) -> np.ndarray:
    """Format nonwear data into an array for saving.

    Args:
        nonwear_data: The nonwear measurement.
        reference_measure: The measure from which the temporal resolution will be taken.
            Any epoch1 measure can be used, but enmo is used for consistency.

    Returns:
        1-D binary np.ndarray, with 1 indicating nonwear. Will be of the same length as
            the timestamps in the reference_measure.

    """
    nonwear_df = pl.DataFrame(
        {
            "nonwear": nonwear_data.measurements.astype(np.int64),
            "time": nonwear_data.time,
        }
    ).set_sorted("time")

    nonwear_upsample = nonwear_df.upsample(time_column="time", every="5s").fill_null(
        strategy="forward"
    )

    end_sequence = pl.repeat(
        nonwear_upsample["nonwear"][-1],
        n=(len(reference_measure.time) - len(nonwear_upsample["time"])),
        dtype=pl.Int64,
        eager=True,
    )
    full_nonwear_array = pl.concat([nonwear_upsample["nonwear"], end_sequence])

    return full_nonwear_array.to_numpy()


def run(
    input: pathlib.Path,
    output: Optional[pathlib.Path] = None,
    settings: config.Settings = config.Settings(),
    calibrator: calibration.Calibration = calibration.Calibration(),
    detect_nonwear_kwargs: Optional[Dict[str, Any]] = None,
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
        detect_nonwear_kwargs: The arguments to the nonwear function delivered as a
            dictionary. Arguements are short_epoch_length, n_short_epoch_in_long_epoch,
            std_criteria, range_criteria.

    Returns:
        All calculated data in a save ready format as a Results object.

    """
    watch_data = readers.read_watch_data(input)
    try:
        calibrated_acceleration = calibrator.run(watch_data.acceleration)
    except Exception as e:
        print("Calibration FAILED:", str(e))
        print("Proceeding without calibration.")
        calibrated_acceleration = watch_data.acceleration
    enmo = metrics.euclidean_norm_minus_one(calibrated_acceleration)
    enmo_epoch1 = computations.moving_mean(enmo)
    anglez = metrics.angle_relative_to_horizontal(calibrated_acceleration)
    anglez_epoch1 = computations.moving_mean(anglez)
    non_wear_array = metrics.detect_nonwear(
        calibrated_acceleration, **(detect_nonwear_kwargs or {})
    )
    sleep_detector = analytics.GGIRSleepDetection(anglez_epoch1)
    sleep_windows = sleep_detector.run_sleep_detection()
    physical_activity_levels = analytics.compute_physical_activty_categories(
        enmo_epoch1,
        (
            settings.LIGHT_THRESHOLD,
            settings.MODERATE_THRESHOLD,
            settings.VIGOROUS_THRESHOLD,
        ),
    )
    sleep_array = models.Measurement(
        measurements=format_sleep_data(
            sleep_windows=sleep_windows, reference_measure=enmo_epoch1
        ),
        time=enmo_epoch1.time,
    )
    nonwear_epoch1 = models.Measurement(
        measurements=format_nonwear_data(
            nonwear_data=non_wear_array, reference_measure=enmo_epoch1
        ),
        time=enmo_epoch1.time,
    )

    results = Results(
        enmo=enmo,
        anglez=anglez,
        enmo_epoch1=enmo_epoch1,
        anglez_epoch1=anglez_epoch1,
        nonwear_array=non_wear_array,
        sleep_windows=sleep_windows,
        physical_activity_levels=physical_activity_levels,
        sleep_windows_epoch1=sleep_array,
        nonwear_epoch1=nonwear_epoch1,
    )
    if output is not None:
        try:
            results.save_results(output=output)
        except (InvalidFileTypeError, FileNotFoundError) as e:
            print(e)

    return results
