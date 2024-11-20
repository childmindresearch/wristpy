"""Handle idle sleep mode special case."""

import numpy as np
import polars as pl

from wristpy.core import models


def impute_idle_sleep_mode_gaps(acceleration: models.Measurement) -> models.Measurement:
    """This function imputes the gaps in the idle sleep mode data.

    This function is called when the idle_sleep_mode_flag is True. It imputes the gaps
    in the acceleration data by assuming the watch is idle in a face up position.
    The acceleration data is filled in at a linear sampling rate, estimated based on the
    first 100 samples timestamps, with (np.finfo(float).eps, np.finfo(float).eps, -1).

    In cases when the sampling rate leads to unevenly spaced samples within one second,
    eg. 30Hz sampling rate has samples spaced at 33333333ns and 33333343ns within one
    second, the entire data set will be resampled at the highest effective sampling rate
    that allows for for linearly spaced samples within one second,
    to nanosecond precision.

    Args:
        acceleration: The raw acceleration data.

    Returns:
        A Measurement object with the modified acceleration data.
    """

    def _find_effective_sampling_rate(sampling_rate: int) -> int:
        """Helper function to find the effective sampling rate.

        This function finds the new sampling rate that allows for linearly spaced
        samples within one second, to nanosecond precision.

        Args:
            sampling_rate: The original sampling rate.

        Returns:
            The new effective sampling rate.
        """
        for effective_sr in range(sampling_rate, 1, -1):
            if 1e9 % (1e9 / effective_sr) == 0:
                return effective_sr
        return 1

    acceleration_polars_df = pl.DataFrame(
        {
            "X": acceleration.measurements[:, 0],
            "Y": acceleration.measurements[:, 1],
            "Z": acceleration.measurements[:, 2],
            "time": acceleration.time,
        }
    )
    fill_value = np.finfo(float).eps
    sampling_space_nanosec = round(
        np.mean(
            acceleration.time[:100]
            .diff()
            .drop_nulls()
            .dt.total_nanoseconds()
            .to_numpy()
            .astype(dtype=float)
        )
    )
    sampling_rate = int(1e9 / sampling_space_nanosec)

    effective_sampling_rate = _find_effective_sampling_rate(sampling_rate)
    effective_sampling_interval = int(1e9 / effective_sampling_rate)

    filled_acceleration = (
        acceleration_polars_df.set_sorted("time")
        .group_by_dynamic("time", every=f"{effective_sampling_interval}ns")
        .agg(pl.exclude("time").mean())
        .upsample("time", every=f"{effective_sampling_interval}ns", maintain_order=True)
        .with_columns(
            pl.col("X").fill_null(value=fill_value),
            pl.col("Y").fill_null(value=fill_value),
            pl.col("Z").fill_null(value=-1),
        )
    )

    return models.Measurement.from_data_frame(filled_acceleration)
