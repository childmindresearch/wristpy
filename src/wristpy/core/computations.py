"""This module will contain functions to compute statistics on the sensor data."""

import polars as pl

from wristpy.core import models


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
    if epoch_length < 1:
        raise ValueError("Epoch length must be greater than 0")

    window_size_seconds = str(epoch_length) + "s"

    measurement_polars_df = pl.concat(
        [pl.DataFrame(array.measurements), pl.DataFrame(array.time)], how="horizontal"
    )

    measurement_polars_df = measurement_polars_df.with_columns(
        pl.col("time").set_sorted()
    )

    take_mean_expression = [
        pl.all().exclude(["time"]).drop_nans().mean(),
    ]

    measurement_df_mean = measurement_polars_df.group_by_dynamic(
        index_column="time", every=window_size_seconds
    ).agg(take_mean_expression)

    measurements_mean_array = measurement_df_mean.drop("time").to_numpy()
    if array.measurements.ndim == 1:
        measurements_mean_array = measurements_mean_array.flatten()

    return models.Measurement(
        measurements=measurements_mean_array,
        time=measurement_df_mean["time"],
    )


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
    if epoch_length < 1:
        raise ValueError("Epoch length must be greater than 0")

    window_size_seconds = str(epoch_length) + "s"

    measurement_polars_df = pl.concat(
        [pl.DataFrame(array.measurements), pl.DataFrame(array.time)], how="horizontal"
    )

    measurement_polars_df = measurement_polars_df.with_columns(
        pl.col("time").set_sorted()
    )

    take_std_expression = [
        pl.all().exclude(["time"]).drop_nans().std(),
    ]

    measurement_df_std = measurement_polars_df.group_by_dynamic(
        index_column="time", every=window_size_seconds
    ).agg(take_std_expression)

    measurements_std_array = measurement_df_std.drop("time").to_numpy()
    if array.measurements.ndim == 1:
        measurements_std_array = measurements_std_array.flatten()

    return models.Measurement(
        measurements=measurements_std_array,
        time=measurement_df_std["time"],
    )


def moving_median(
    acceleration: models.Measurement, window_size: int
) -> models.Measurement:
    """Applies moving median to acceleration data.

    Step size for the window is hard-coded to 1 sample.

    Args:
        acceleration: the three dimensional accelerometer data. A Measurement object,
        it will have two attributes. 1) measurements, containing the three dimensional
        accelerometer data in an np.array and 2) time, a pl.Series containing
        datetime.datetime objects.

        window_size: Size of the moving median window. Window is centered.
        Measured in seconds.


    Returns:
        Measurement object with rolling median applied to the measurement data. The
        measurements data will retain it's shape, and the time data will be returned
        unaltered.
    """
    measurements_polars_df = pl.concat(
        [
            pl.DataFrame(acceleration.measurements),
            pl.DataFrame({"time": acceleration.time}),
        ],
        how="horizontal",
    )
    measurements_polars_df = measurements_polars_df.set_sorted("time")
    offset = -((window_size // 2) + 1)
    offset_str = str(offset) + "s"
    moving_median_df = measurements_polars_df.select(
        [
            pl.median("*").rolling(
                index_column="time", period=f"{window_size}s", offset=offset_str
            )
        ]
    )

    return models.Measurement(
        measurements=moving_median_df.drop("time").to_numpy(),
        time=measurements_polars_df["time"],
    )
