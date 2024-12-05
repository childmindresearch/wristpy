"""This module will contain functions to compute statistics on the sensor data."""

import datetime
from typing import Literal

import numpy as np
import polars as pl

from wristpy.core import models


def _moving(
    measurement: models.Measurement,
    epoch_length: int,
    aggregation: Literal["mean", "std", "median"],
    *,
    centered: bool = False,
    continuous: bool = False,
) -> models.Measurement:
    """Internal handler of rolling window functions.

    Args:
        measurement: The measurement to apply a rolling function to.
        epoch_length: Length of the window in seconds.
        aggregation: Name of the function to apply, either 'mean', 'std', or 'median'.
        centered: If true, centers the window. Defaults to False.
        continuous: If true, applies the window to every measurement. If false,
            groups measurements into chunks of epoch_length. Defaults to False.

    Returns:
        The measurement with the rolling function applied to it.
    """
    if epoch_length <= 0:
        raise ValueError("Epoch length must be greater than 0")

    window_size = f"{int(epoch_length * 1e9)}ns"
    if centered:
        offset = f"{-int((epoch_length // 2 + 1) * 1e9)}ns"
    else:
        offset = "0ns"

    if continuous:
        aggregator = getattr(pl, aggregation)("*").rolling(
            index_column="time", period=window_size, offset=offset
        )
        aggregated_df = (
            measurement.lazy_frame()
            .select([aggregator])
            .with_columns(time=measurement.time)
        )
    else:
        aggregator = getattr(pl.exclude(["time"]).drop_nans(), aggregation)
        aggregated_df = (
            measurement.lazy_frame()
            .group_by_dynamic("time", every=window_size, offset=offset)
            .agg(aggregator())
        )

    return models.Measurement.from_data_frame(aggregated_df.collect())


def moving_mean(array: models.Measurement, epoch_length: int = 5) -> models.Measurement:
    """Calculate the moving mean of the sensor data in array.

    Args:
        array: The Measurement object with the sensor data we want to take the mean of
        epoch_length: The length, in seconds, of the window.

    Returns:
        The moving mean of the array in a new Measurement instance.

    Raises:
        ValueError: If the epoch length is not an integer or is less than 1.
    """
    return _moving(array, epoch_length, "mean")


def moving_std(array: models.Measurement, epoch_length: int = 5) -> models.Measurement:
    """Calculate the moving standard deviation (std) of the sensor data in array.

    Args:
        array: The Measurement object with the sensor data we want to take the std of
        epoch_length: The length, in seconds, of the window.

    Returns:
        The moving std of the array in a new Measurement instance.

    Raises:
        ValueError: If the epoch length is less than 1.
    """
    return _moving(array, epoch_length, "std")


def moving_median(array: models.Measurement, epoch_length: int) -> models.Measurement:
    """Applies moving median to acceleration data.

    Step size for the window is hard-coded to 1 sample.

    Args:
        array: The Measurement object with the sensor data we want to take the median
            of.
        epoch_length: Size of the moving median window. Window is centered.
            Measured in seconds.

    Returns:
        Measurement object with rolling median applied to the measurement data. The
        measurements data will retain it's shape, and the time data will be returned
        unaltered.
    """
    return _moving(array, epoch_length, "median", centered=True, continuous=True)


def majority_vote_non_wear(
    nonwear_ggir: models.Measurement,
    nonwear_cta: models.Measurement,
    nonwear_detach: models.Measurement,
    temporal_resolution: float = 60.0,
) -> models.Measurement:
    """This function applies a majority vote on the three possible nonwear outputs.

    The three algorithms are resampled to the same sampling rate, and a majority vote
    is taken at each time point to determine the new nonwear time.


    This assumes the nonwear measurements have the same ending time stamp.

    Args:
        nonwear_ggir: The nonwear algorithm output from the GGIR algorithm.
        nonwear_cta: The nonwear algorithm output from the CTA algorithm.
        nonwear_detach: The nonwear algorithm output from the detach algorithm.
        temporal_resolution: The temporal resolution of the output, in seconds.
            Defaults to 5.0.

    Returns:
        A new Measurement instance at a new temporal resolution.
    """
    min_start_time = min(
        [nonwear_ggir.time[0], nonwear_cta.time[0], nonwear_detach.time[0]]
    )

    max_end_time = max(
        [nonwear_ggir.time[-1], nonwear_cta.time[-1], nonwear_detach.time[-1]]
    )

    nonwear_ggir = _time_fix(nonwear_ggir, max_end_time, min_start_time)
    nonwear_cta = _time_fix(nonwear_cta, max_end_time, min_start_time)
    nonwear_detach = _time_fix(nonwear_detach, max_end_time, min_start_time)

    nonwear_ggir = resample(nonwear_ggir, temporal_resolution)
    nonwear_cta = resample(nonwear_cta, temporal_resolution)
    nonwear_detach = resample(nonwear_detach, temporal_resolution)

    nonwear_ggir.measurements = np.where(nonwear_ggir.measurements >= 0.5, 1, 0)
    nonwear_cta.measurements = np.where(nonwear_cta.measurements >= 0.5, 1, 0)

    nonwear_value = np.where(
        (
            nonwear_ggir.measurements
            + nonwear_cta.measurements
            + nonwear_detach.measurements
        )
        >= 2,
        1,
        0,
    )

    return models.Measurement(measurements=nonwear_value, time=nonwear_ggir.time)


def _time_fix(
    nonwear: models.Measurement,
    max_end_time: datetime.datetime,
    min_start_time: datetime.datetime,
) -> models.Measurement:
    """Helper function to fix the time of the nonwear measurements."""
    if nonwear.time[0] > min_start_time:
        nonwear.time = pl.concat(
            [pl.Series([min_start_time], dtype=pl.Datetime("ns")), nonwear.time]
        )
        nonwear.measurements = np.append(nonwear.measurements[0], nonwear.measurements)

    if nonwear.time[-1] < max_end_time:
        nonwear.time.append(pl.Series([max_end_time], dtype=pl.Datetime("ns")))
        nonwear.measurements = np.append(nonwear.measurements, nonwear.measurements[-1])
    return nonwear


def resample(measurement: models.Measurement, delta_t: float) -> models.Measurement:
    """Resamples a measurement to a different timescale.

    Args:
        measurement: The measurement to resample.
        delta_t: The new time step, in seconds. This will be rounded to the nearest
            nanosecond.

    Returns:
        The resampled measurement.

    Raises:
        NotImplementedError: Raised for measurements with non-monotonically increasing
            time.
    """
    if delta_t <= 0:
        msg = "delta_t must be positive."
        raise ValueError(msg)

    all_delta_t = (measurement.time[1:] - measurement.time[:-1]).unique()

    n_nanoseconds_in_second = 1_000_000_000
    current_delta_t = all_delta_t[0].seconds * n_nanoseconds_in_second
    requested_delta_t = round(delta_t * n_nanoseconds_in_second)

    if current_delta_t == requested_delta_t:
        return measurement

    measurement_df = (
        pl.from_numpy(measurement.measurements)
        .with_columns(time=measurement.time)
        .set_sorted("time")
    )

    if current_delta_t > requested_delta_t:
        resampled_df = (
            measurement_df.upsample(
                time_column="time", every=f"{requested_delta_t}ns", maintain_order=True
            )
            .interpolate()
            .fill_null("forward")
        )
    else:
        resampled_df = measurement_df.group_by_dynamic(
            "time", every=f"{requested_delta_t}ns"
        ).agg(pl.exclude("time").mean())

    new_measurement = (
        resampled_df.drop("time").to_numpy().reshape((len(resampled_df), -1)).squeeze()
    )
    return models.Measurement(
        measurements=new_measurement,
        time=resampled_df["time"],
    )
