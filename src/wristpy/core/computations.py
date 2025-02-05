"""This module will contain functions to compute statistics on the sensor data."""

from typing import Literal

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


def resample(measurement: models.Measurement, delta_t: float) -> models.Measurement:
    """Resamples a measurement to a different timescale.

    Args:
        measurement: The measurement to resample.
        delta_t: The new time step, in seconds. This will be rounded to the nearest
            nanosecond.

    Returns:
        The resampled measurement.

    Raises:
        ValueError: Raised for negative delta_t.
    """
    if delta_t <= 0:
        msg = "delta_t must be positive."
        raise ValueError(msg)

    all_delta_t = (measurement.time[1:] - measurement.time[:-1]).unique()

    n_nanoseconds_in_second = 1_000_000_000
    current_delta_t = all_delta_t[0].seconds * n_nanoseconds_in_second
    requested_delta_t = round(delta_t * n_nanoseconds_in_second)

    measurement_df = (
        pl.from_numpy(measurement.measurements)
        .with_columns(time=measurement.time)
        .set_sorted("time")
    )

    if current_delta_t >= requested_delta_t:
        resampled_df = (
            measurement_df.upsample(
                time_column="time",
                every=f"{requested_delta_t}ns",
                maintain_order=True,
                start_by="datapoint",
            )
            .interpolate()
            .fill_null("forward")
        )
    else:
        resampled_df = measurement_df.group_by_dynamic(
            "time", every=f"{requested_delta_t}ns", start_by="datapoint"
        ).agg(pl.exclude("time").mean())

    new_measurement = (
        resampled_df.drop("time").to_numpy().reshape((len(resampled_df), -1)).squeeze()
    )
    return models.Measurement(
        measurements=new_measurement,
        time=resampled_df["time"],
    )
