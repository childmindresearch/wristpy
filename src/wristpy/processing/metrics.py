"""Calculate base metrics, enmo, anglez, and non-wear detection."""

import polars as pl
import warnings

from wristpy.core.models import Measurement


def moving_mean(array: Measurement, epoch_length: int = 5) -> Measurement:
    """Calculate the moving mean of the sensor data in array.

    The input must be a Measurement instance. The window size must be a posititve
    non-zero integer. The function converts the epoch_length to a string that can
    be used by polars group_by_dynamic. We then create a polars dataframe that is
    a concatenation of the sensor data and the equivalent time data. We group
    the data by creating a windowbased on the timestamp using the "every" function
    and then take the mean of that group.

    Args:
        array: The Measurement object with the sensor data we want to take the mean of
        epoch_length: The length of the window to calculate the moving mean over.

    Returns:
        The moving mean of the array in a new Measurement instance.
    """
    if not isinstance(array, Measurement) or array.measurements.size == 0:
        warnings.warn(
            "Input is not a Measurement or is an empty Measurement. "
            "Returning the input."
        )
        return array
    if not isinstance(epoch_length, int):
        raise ValueError("Epoch length must be an integer")
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

    return Measurement(
        measurements=measurement_df_mean.drop("time").to_numpy(),
        time=measurement_df_mean["time"],
    )
