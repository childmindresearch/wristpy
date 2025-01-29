"""Calculate base metrics, anglez and enmo."""

from typing import Tuple

import numpy as np
import polars as pl
from scipy import interpolate, stats

from wristpy.core import config, models

logger = config.get_logger()


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
    n_short_epoch_in_long_epoch: int = 4,
    std_criteria: float = 0.013,
) -> models.Measurement:
    """Set non_wear_flag based on accelerometer data.

    This implements a modified version of the GGIR "2023" non-wear detection algorithm.
    Briefly, the algorithm, creates a sliding window of long epoch length that steps
    forward by the short epoch length. The long epoch length is an integer multiple of
    the short epoch length, that can be specified by the user.
    It checks if the acceleration data in that long window, for each axis, meets the
    criteria threshold for the standard deviation of acceleration values to
    compute a non-wear value. The total non-wear value (0, 1, 2, 3) for the long window
    is the sum of each axis.
    The non-wear value is applied to all the short windows that make up the long
    window. Additionally, as the majority of the short windows are part of multiple long
    windows, the value of a short window is updated to the maximum nonwear value from
    these overlaps.
    Finally, there is a pass to find isolated "1s" in the non-wear
    value, and set them to 2 if surrounded by > 1 values. The non-wear flag is set to
    1 (true) if the non-wear value is >= 2, and 0 (false) otherwise.


    Args:
        acceleration: The Measurment instance that contains the calibrated acceleration
            data.
        short_epoch_length: The short window size, in seconds.
        n_short_epoch_in_long_epoch: Number of short epochs that makeup one long epoch.
        std_criteria: Threshold criteria for standard deviation.


    Returns:
        A new Measurment instance with the non-wear flag and corresponding timestamps.
    """
    logger.debug("Detecting non-wear data.")
    acceleration_grouped_by_short_window = _group_acceleration_data_by_time(
        acceleration, short_epoch_length
    )

    nonwear_value_array = _compute_nonwear_value_array(
        acceleration_grouped_by_short_window,
        n_short_epoch_in_long_epoch,
        std_criteria,
    )

    nonwear_value_array_cleaned = _cleanup_isolated_ones_nonwear_value(
        nonwear_value_array
    )
    non_wear_flag = np.where(nonwear_value_array_cleaned >= 2, 1, 0)

    return models.Measurement(
        measurements=non_wear_flag, time=acceleration_grouped_by_short_window["time"]
    )


def _group_acceleration_data_by_time(
    acceleration: models.Measurement, window_length: int
) -> pl.DataFrame:
    """Helper function to group the acceleration data by short windows.

    Args:
        acceleration: The Measurment instance that contains the calibrated acceleration.
        window_length: The window size, in seconds.

    Returns:
        A polars DataFrame with the acceleration data grouped by window_length.
    """
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

    acceleration_grouped_by_window_length = acceleration_polars_df.group_by_dynamic(
        index_column="time", every=(str(window_length) + "s")
    ).agg([pl.all().exclude(["time"])])

    return acceleration_grouped_by_window_length


def _compute_nonwear_value_array(
    grouped_acceleration: pl.DataFrame,
    n_short_epoch_in_long_epoch: int,
    std_criteria: float,
) -> np.ndarray:
    """Helper function to calculate the nonwear value array.

    This function calculates the nonwear value array based on the GGIR 2023 methodology.
    It computes the nonwear value for each axis, based on the acceleration data that
    makes up one long epoch window. That nonwear value is then applied to all the
    short windows that make up the long window. It iterates forward by one short_window
    length and repeats the process. For the overlapping short windows, the maximum
    nonwear value is kept and is assigned to the nonwear value array.

    Args:
        grouped_acceleration: The acceleration data grouped into short windows.
        n_short_epoch_in_long_epoch: Number of short epochs that makeup one long epoch.
        std_criteria: Threshold criteria for standard deviation.

    Returns:
        Non-wear value array.
    """
    total_n_short_windows = len(grouped_acceleration)
    nonwear_value_array = np.zeros(total_n_short_windows)

    for window_n in range(total_n_short_windows - n_short_epoch_in_long_epoch + 1):
        acceleration_selected_long_window = grouped_acceleration[
            window_n : window_n + n_short_epoch_in_long_epoch
        ]

        calculated_nonwear_value = acceleration_selected_long_window.select(
            pl.col("X", "Y", "Z").map_batches(
                lambda df: _compute_nonwear_value_per_axis(
                    df,
                    std_criteria,
                )
            )
        ).sum_horizontal()

        max_window_value = np.maximum(
            nonwear_value_array[window_n : window_n + n_short_epoch_in_long_epoch],
            np.repeat(calculated_nonwear_value, n_short_epoch_in_long_epoch),
        )
        nonwear_value_array[window_n : window_n + n_short_epoch_in_long_epoch] = (
            max_window_value
        )

    return nonwear_value_array


def _compute_nonwear_value_per_axis(
    axis_acceleration_data: pl.Series,
    std_criteria: float,
) -> bool:
    """Helper function to calculate the nonwear criteria per axis.

    Args:
        axis_acceleration_data: The long window acceleration data for one axis.
            It is a pl.Series chunked into short windows where each row is a list of the
            acceleration data of one axis (length of each list is the number of samples
            that make up short_epoch_length in seconds).
        std_criteria: Threshold criteria for standard deviation


    Returns:
        Non-wear value for the axis.
    """
    axis_long_window_data = pl.concat(axis_acceleration_data, how="vertical")
    axis_std = axis_long_window_data.std()
    criteria_boolean = axis_std < std_criteria

    return criteria_boolean


def _cleanup_isolated_ones_nonwear_value(nonwear_value_array: np.ndarray) -> np.ndarray:
    """Helper function to cleanup isolated ones in nonwear value array.

    This function finds isolated ones in the nonwear value array and
    sets them to 2 if they are surrounded by values > 1.

    Args:
        nonwear_value_array: The nonwear value array that needs to be cleaned up.
            It is a 1D numpy array.

    Returns:
        The modified nonwear value array.
    """
    nonwear_value_array = nonwear_value_array.astype(int)

    left_neighbors = np.roll(nonwear_value_array, 1)
    right_neighbors = np.roll(nonwear_value_array, -1)

    condition = (left_neighbors > 1) & (right_neighbors > 1) & nonwear_value_array == 1
    condition[0] = False
    condition[-1] = False

    nonwear_value_array[condition] = 2

    return nonwear_value_array


def interpolate_measure(
    acceleration: models.Measurement, new_frequency: int = 100
) -> models.Measurement:
    """Interpolate the measure to a new sampling rate using natural cubic spline.

    Args:
        acceleration: Accelerometer data and associated timestamps.
        new_frequency: The new frequency the measure will be interpolated to in Hz. For
            the purposes of the MIMS algorithm defaults to 100Hz.

    Returns:
        A Measurement object with interpolated acceleration data.
    """
    epoch_time_seconds = acceleration.time.dt.epoch(time_unit="ns").to_numpy() / 1e9
    start_time = epoch_time_seconds[0]
    end_time = epoch_time_seconds[-1]

    duration_s = end_time - start_time
    n_points = int(duration_s * new_frequency) + 1

    interpolated_time = np.linspace(start_time, end_time, n_points, endpoint=True)
    interpolated_data = np.zeros((len(interpolated_time), 3))

    for axis in range(3):
        cubic_spline = interpolate.CubicSpline(
            epoch_time_seconds, acceleration.measurements[:, axis], bc_type="natural"
        )
        interpolated_data[:, axis] = cubic_spline(interpolated_time)

    new_time_ns = (interpolated_time * 1e9).astype(np.int64)
    new_time_series = pl.Series(
        "interpolated_time", new_time_ns, dtype=pl.Datetime("ns")
    )

    return models.Measurement(measurements=interpolated_data, time=new_time_series)


def extrapolate_points(
    acceleration: models.Measurement,
    dynamic_range: Tuple[float, float],
    noise_std: float = 0.3,
    num_std: int = 5,
    smoothing: float = 0.6,
    scale: float = 1,
    probability: float = 0.95,
) -> None:
    """Identify maxed out values, and extrapolate points.

    Args:
        acceleration: Acceleration data that has been interpolated to 100Hz.
        dynamic_range: Dynamic range of device used.
        noise_level: Device noise level
        num_std: How many standard deviations,
        smoothing: Smoothing parameter for spline function. A value of 0 would fit the
            points linearly, a value of 1 would fit every point.
        scale: theta value, scale parameter for gamma dist.
        probability = probability threshold for finding k(Shape) parameter.

    Returns:
        Acceleration data with extrapolated points.
    """
    maxed_out_likelihood = _find_maxed_out_likelihood(
        acceleration=acceleration,
        dynamic_range=dynamic_range,
        num_std=num_std,
        noise_std=noise_std,
        scale=scale,
    )

    return None


def _find_maxed_out_likelihood(
    dynamic_range: Tuple[float, float],
    num_std: int,
    noise_std: float,
    acceleration: models.Measurement,
    scale: float = 1,
    probability: float = 0.95,
) -> np.ndarray:
    """Find probability values are maxed out."""
    range_max = max(dynamic_range)
    buffer_threshold = range_max - num_std * noise_std
    distance = np.abs(acceleration.measurements) - buffer_threshold

    k = _brute_force_k(
        target_probability=probability,
        step=0.001,
        standard_deviation=num_std * noise_std,
    )

    raw_likelihood = stats.gamma.cdf(distance, k, scale=scale)
    return np.where(distance < 0, 0, raw_likelihood)


def _brute_force_k(
    standard_deviation: float,
    target_probability: float = 0.95,
    step: float = 0.001,
    scale: float = 1.0,
) -> float:
    """Find the shape value for gamma distribution."""
    k_values = np.arange(0.5, 0.001, -step)
    previous_probability = 1.0
    previous_k = 0
    result = 0

    for k in k_values:
        current_probability = stats.gamma.cdf(standard_deviation, a=k, scale=scale)

        if (
            previous_probability < target_probability
            and current_probability >= target_probability
        ):
            if abs(target_probability - previous_probability) > abs(
                current_probability - target_probability
            ):
                result = k
            else:
                result = previous_k
            break
        previous_probability = current_probability
        previous_k = k

    return result
