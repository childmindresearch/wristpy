"""This module will contain functions to compute statistics on the sensor data."""

import polars as pl

from wristpy.core import config, models

logger = config.get_logger()


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
    return resample(array, epoch_length)


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
        [pl.DataFrame(acceleration.measurements), pl.DataFrame(acceleration.time)],
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
    if len(all_delta_t) != 1:
        msg = " ".join(
            [
                "Resampling function only accepts measurements with monotonically"
                "increasing time."
            ]
        )
        raise NotImplementedError(msg)

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


def _measurement_to_dataframe(measurement: models.Measurement) -> pl.DataFrame:
    return pl.concat(
        [pl.DataFrame(measurement.measurements), pl.DataFrame(measurement.time)],
        how="horizontal",
    )
