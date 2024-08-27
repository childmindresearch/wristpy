"""Python based runner."""

import pathlib
from typing import Dict, List, Optional

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
    """dataclass containing results of run."""

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
        """Convert to polars and save the dataframe as a csv or parquet file."""
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
        """Format results into a dataframe."""
        time_data = self.enmo_epoch1.time if use_epoch1_time else self.enmo.time

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
    sleep_windows: List[analytics.SleepWindow], epoch1_measure: models.Measurement
) -> np.ndarray:
    """Formats sleep windows into an array for saving."""
    sleep_array = np.zeros(len(epoch1_measure.time))

    for window in sleep_windows:
        sleep_mask = (epoch1_measure.time >= window.onset) & (
            epoch1_measure.time <= window.wakeup
        )
        sleep_array[sleep_mask] = 1

    return sleep_array


def format_nonwear_data(
    nonwear_data: models.Measurement, epoch1_measure: models.Measurement
) -> np.ndarray:
    """Format nonwear data into an array for saving."""
    nonwear_df = pl.DataFrame(
        {
            "nonwear": nonwear_data.measurements,
            "time": nonwear_data.time,
        }
    )
    nonwear_upsample = nonwear_df.upsample(time_column="time", every="5s").fill_null(
        strategy="forward"
    )

    end_sequence = pl.repeat(
        nonwear_upsample["nonwear"][-1],
        n=(len(epoch1_measure.time) - len(nonwear_upsample["time"])),
        dtype=pl.Int64,
        eager=True,
    )
    full_nonwear_array = pl.concat([nonwear_upsample["nonwear"], end_sequence])

    return full_nonwear_array.to_numpy()


def run(
    input: pathlib.Path,
    output: pathlib.Path | None = None,
    settings: config.Settings = config.Settings(),
    calibrator: calibration.Calibration = calibration.Calibration(),
    detect_nonwear_kwargs: Optional[Dict[str, float]] = None,
) -> Results:
    """Runs wristpy."""
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
            sleep_windows=sleep_windows, epoch1_measure=enmo_epoch1
        ),
        time=enmo_epoch1.time,
    )
    nonwear_epoch1 = models.Measurement(
        measurements=format_nonwear_data(
            nonwear_data=non_wear_array, epoch1_measure=enmo_epoch1
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
