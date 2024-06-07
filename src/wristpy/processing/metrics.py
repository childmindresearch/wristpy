"""Calculate base metrics, anglez and enmo."""

import numpy as np
import polars as pl

from wristpy.core import models


def euclidean_norm_minus_one(acceleration: models.Measurement) -> models.Measurement:
    """Compute ENMO, the Euclidean Norm Minus One (1 standard gravity unit).

    Negative values of ENMO are set to zero because ENMO is meant as a measure of
    physival activity. Negative values steming from imperfect calibration or noise from
    the device would have no meaningful interpretation in this context and would be
    detrimental to the intended analysis.

    Args:
        acceleration: the three dimensional accelerometer data. A Measurement object,
        it will have two attributes. 1) measurements, containing the three dimensional
        accelerometer data in an np.array and 2) time, a pl.Series containing
        datetime.datetime objects.

    Returns:
        A Measurement object containing the calculated ENMO values and the
        associated time stamps taken from the input.
    """
    enmo = np.linalg.norm(acceleration.measurements, axis=1) - 1

    enmo = np.maximum(enmo, 0)

    return models.Measurement(measurements=enmo, time=acceleration.time)


def angle_relative_to_horizontal(
    acceleration: models.Measurement,
) -> models.Measurement:
    """Calculate the angle of the acceleration vector relative to the horizontal plane.

    Args:
        acceleration: the three dimensional accelerometer data. A Measurement object,
        it will have two attributes. 1) measurements, containing the three dimensional
        accelerometer data in an np.array and 2) time, a pl.Series containing
        datetime.datetime objects.

    Returns:
        A Measurement instance containing the values of the angle relative to the
        horizontal plane and the associated timestamps taken from the input unaltered.
        The angle is measured in degrees.
    """
    xy_projection_magnitute = np.linalg.norm(acceleration.measurements[:, 0:2], axis=1)

    angle_radians = np.arctan(acceleration.measurements[:, 2] / xy_projection_magnitute)

    angle_degrees = np.degrees(angle_radians)

    return models.Measurement(measurements=angle_degrees, time=acceleration.time)


def detect_nonwear(
    acceleration: models.Measurement,
    short_epoch_length: int = 900,
    long_epoch_length: int = 3600,
    std_criteria: float = 0.013,
    range_criteria: float = 0.05,
) -> models.Measurement:
    """Set non_wear_flag based on accelerometer data.

    This implements GGIR "2023" non-wear detection algorithm.
    Briefly, the algorithm, creates a sliding window of long epoech length that steps
    forward by the short epoch length.
    It checks if the acceleration data in that long window, for each axis, meets certain
    criteria thresholds to compute a non-wear value.
    And then applies that non-wear value to all the short windows that make up the long
    window. Additionally, for overlapping windows, the maximum of the non-wear value for
    the overlaps is kept. Finally, there is a pass to find isolated "1s" in the non-wear
    value, and set them to 2 if surrounded by > 1 values. The non-wear flag is set to
    1 (true) if the non-wear value is >= 2, and 0 (false) otherwise.


    Args:
        acceleration: The Measurment instance that contains the calibrated acceleration
        data
        short_epoch_length: The short window size, in seconds, for non-wear detection
        long_epoch_length: The long window size, in seconds, for non-wear detection
        std_criteria: Threshold criteria for standard deviation
        range_criteria: Threshold criteria for range of acceleration


    Returns:
        A new Measurment instance with the non-wear flag and corresponding timestamps.
    """
    n_short_epoch_in_long_epoch = int(long_epoch_length / short_epoch_length)

    acceleration_polars_df = pl.DataFrame(
        {
            "X": acceleration.measurements[:, 0],
            "Y": acceleration.measurements[:, 1],
            "Z": acceleration.measurements[:, 2],
            "time": acceleration.time,
        }
    )
    acceleration_polars_df = acceleration_polars_df.with_columns(
        pl.col("time").set_sorted()
    )

    acceleration_grouped_by_short_window = acceleration_polars_df.group_by_dynamic(
        index_column="time", every=(str(short_epoch_length) + "s")
    ).agg([pl.all().exclude(["time"])])

    nonwear_value_array = np.zeros(len(acceleration_grouped_by_short_window))

    for window_n in range(
        len(acceleration_grouped_by_short_window) - n_short_epoch_in_long_epoch + 1
    ):
        acceleration_selected_long_window = acceleration_grouped_by_short_window[
            window_n : window_n + n_short_epoch_in_long_epoch
        ]

        calculated_nonwear_value = acceleration_selected_long_window.select(
            pl.col("X", "Y", "Z").map_batches(
                lambda df: _compute_nonwear_value_per_axis(
                    df, std_criteria, range_criteria
                )
            )
        ).sum_horizontal()

        max_window_value = np.maximum(
            nonwear_value_array[window_n : window_n + n_short_epoch_in_long_epoch],
            np.repeat(calculated_nonwear_value, n_short_epoch_in_long_epoch),
        )
        nonwear_value_array[
            window_n : window_n + n_short_epoch_in_long_epoch
        ] = max_window_value

    nonwear_ones = np.where(nonwear_value_array == 1)[0]

    for ones_index in nonwear_ones:
        if ones_index == 0:
            continue
        if ones_index == len(nonwear_value_array) - 1:
            continue
        if (nonwear_value_array[ones_index - 1] > 1) and (
            nonwear_value_array[ones_index + 1] > 1
        ):
            nonwear_value_array[ones_index] = 2

    non_wear_flag = np.where(nonwear_value_array >= 2, 1, 0)

    return models.Measurement(
        measurements=non_wear_flag, time=acceleration_grouped_by_short_window["time"]
    )


def _compute_nonwear_value_per_axis(
    axis_acceleration_data: pl.Series, std_criteria: float, range_criteria: float
) -> int:
    """Helper function to calculate the nonwear criteria per axis.

    Args:
        axis_acceleration_data: The long window acceleration data for one axis.
        It is polars.Series chunked into short windows where each row is a list of the
        acceleration data of one axis (length of each list is the number of samples
        that make up short_epoch_length in seconds).
        std_criteria: Threshold criteria for standard deviation
        range_criteria: Threshold criteria for range of acceleration

    Returns:
        Non-wear value for the axis.
    """
    axis_long_window_data = pl.concat(axis_acceleration_data, how="vertical")
    axis_std = axis_long_window_data.std()
    axis_range = axis_long_window_data.max() - axis_long_window_data.min()

    criteria_boolean = (axis_std < std_criteria) & (axis_range < range_criteria)
    return int(criteria_boolean)
