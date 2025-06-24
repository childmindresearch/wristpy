"""Calculate Monitor Indepdent Movement Summary Units."""

from typing import List, Literal, Optional, Tuple

import numpy as np
import polars as pl
from scipy import integrate, interpolate, signal, stats

from wristpy.core import config, models

logger = config.get_logger()


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
    dynamic_range: Tuple[float, float] = (-8.0, 8.0),
    noise: float = 0.03,
    smoothing: float = 0.6,
    scale: float = 1,
    target_probability: float = 0.95,
    confidence_threshold: float = 0.5,
    neighborhood_size: float = 0.05,
    sampling_rate: int = 100,
) -> models.Measurement:
    """Identify maxed out values, and extrapolate points.

    Args:
        acceleration: Acceleration data that has been interpolated to 100Hz.
        dynamic_range: Dynamic range of device used. This information can be gathered
            from device metadata.
        noise: Device noise level.
        smoothing: Smoothing parameter for spline function. A value of 0 fits the data
            exactly, higher values will fit the data more smoothly.
        scale: Theta value, scale parameter for gamma distribution.
        target_probability: Threshold used in determining the shape parameter (k) of the
            gamma distribution. Probability that a value 3 standard deviations from the
            buffer zone is maxed-out.
        confidence_threshold: Threshold for what constitutes a major jump or drop in
            value.
        neighborhood_size: Duration of neighborhood in seconds. This parameter
            determines how much data on each side of the maxed-out region is used
            for fitting the local regression model. The parameter is used to calculate
            the number of points to create in the oversampled data (n_over).
        sampling_rate: The sampling rate in Hz. Assumed to take place after
            interpolation so it will typically be 100.

    Returns:
        A Measurement object with maxed out regions replaced with extrapolated values.

    References:
        John, D., Tang, Q., Albinali, F. and Intille, S., 2019. An Open-Source
            Monitor-Independent Movement Summary for Accelerometer Data Processing.
            Journal for the Measurement of Physical Behaviour, 2(4), pp.268-281.
    """
    time_numeric = acceleration.time.dt.epoch(time_unit="ns").to_numpy()
    extrapolated_axes = []
    for axis in acceleration.measurements.T:
        marker = _find_markers(
            axis=axis,
            dynamic_range=dynamic_range,
            noise=noise,
            scale=scale,
            target_probability=target_probability,
        )
        neighbors = _extrapolate_neighbors(
            marker=marker,
            confidence_threshold=confidence_threshold,
            neighborhood_size=neighborhood_size,
            sampling_rate=sampling_rate,
        )
        points = _extrapolate_fit(
            axis=axis,
            time_numeric=time_numeric,
            marker=marker,
            neighbors=neighbors,
            smoothing=smoothing,
            neighborhood_size=neighborhood_size,
            sampling_rate=sampling_rate,
        )
        extrapolated_values = _extrapolate_interpolate(
            axis=axis,
            time_numeric=time_numeric,
            marker=marker,
            points=points,
            confidence_threshold=confidence_threshold,
        )
        extrapolated_axes.append(extrapolated_values)

    extrapolated_acceleration = np.column_stack(extrapolated_axes)

    return models.Measurement(
        measurements=extrapolated_acceleration, time=acceleration.time
    )


def _find_markers(
    axis: np.ndarray,
    dynamic_range: Tuple[float, float],
    noise: float = 0.03,
    scale: float = 1.0,
    target_probability: float = 0.95,
) -> np.ndarray:
    """Determine the likelihood that each sample is near the limits of the device range.

    Args:
        axis: Acceleration data along one axis.
        dynamic_range: Tuple of floats indicating the device's value range. This
            information can be gathered from watch metadata.
        noise: Typical noise value for device.
        scale: Scale value (theta) for gamma distribution. Typically set at 1.
        target_probability: Threshold used in determing the shape parameter (k) of the
            gamma distribution. Probability that a value 3 standard deviations from the
            buffer zone is maxed-out.

    Returns:
        Array indicating the likelihood that corresponding value in the axis data is
        'maxed-out'.
    """
    positive_buffer_zone = max(dynamic_range) - 5 * (noise + 1e-5)
    negative_buffer_zone = min(dynamic_range) + 5 * (noise + 1e-5)
    shape_k = _brute_force_k(
        standard_deviation=3 * (noise + 1e-5),
        scale=scale,
        target_probability=target_probability,
    )
    marker = np.zeros(len(axis))
    positive_idx = axis >= 0
    negative_idx = axis < 0

    marker[positive_idx] = stats.gamma.cdf(
        axis[positive_idx] - positive_buffer_zone, a=shape_k, scale=scale
    )
    marker[negative_idx] = -stats.gamma.cdf(
        -axis[negative_idx] + negative_buffer_zone, a=shape_k, scale=scale
    )

    return marker


def _brute_force_k(
    standard_deviation: float,
    k_max: float = 0.5,
    k_min: float = 0.001,
    k_step: float = 0.001,
    scale: float = 1.0,
    target_probability: float = 0.95,
) -> float:
    """Find the shape parameter(k) for the gamma distribution.

    Args:
        standard_deviation: The point at which to evaluate the gamma cumulative
            distribution function, should be 3 times the noise.
        k_max: Maximum value for the shape parameter search range.
            Default is 0.5.
        k_min: Minimum value for the shape parameter search range.
            Default is 0.001.
        k_step: Step size for the shape parameter search.
        scale: Scale value (theta) for gamma distribution. Typically set at 1.
        target_probability: Threshold used in determining the shape parameter (k) of the
            gamma distribution. Probability that a value 3 standard deviations from the
            buffer zone is maxed-out.
        scale: Scale value (theta) for gamma distribution. Typically set at 1.

    Returns:
        The optimal shape parameter that makes the gamma CDF at the given standard
        deviation approximately equal to the target probability.
    """
    k_values = np.arange(k_max, k_min, -k_step)
    previous_probability = 1.0
    previous_k = 0.0
    result = 0.0

    for k in k_values:
        current_probability = stats.gamma.cdf(standard_deviation, a=k, scale=scale)

        if (
            previous_probability < target_probability
            and current_probability >= target_probability
        ):
            if abs(target_probability - previous_probability) > abs(
                current_probability - target_probability
            ):
                result = float(k)
            else:
                result = previous_k
            return result

        previous_probability = current_probability
        previous_k = float(k)

    return result


def _extrapolate_neighbors(
    marker: np.ndarray,
    confidence_threshold: float = 0.5,
    neighborhood_size: float = 0.05,
    sampling_rate: int = 100,
) -> pl.DataFrame:
    """Find neighborhoods around maxed-out regions for regression fitting.

    Args:
        marker: Array containing values that mark the probability that each sample is
            maxed-out. Values close to ±1 indicate high likelihood that the value is
            near the limit of the dynamic range, with the sign indicating if it's near
            the upper bound, or lower bound. Values close to zero indicate low
            likelihood of being near range limit.
        confidence_threshold: Threshold for what constitutes a significant change in
            marker values.
        neighborhood_size: Duration of neighborhood in seconds to use around each
            maxed-out region for regression fitting. Default is 0.05 (50ms).
        sampling_rate: The sampling rate of the data in Hz. Default is 100Hz as data
            should be interpolated to 100hz prior to this step.

    Returns:
        DataFrame containing the indices of maxed-out regions (start, end) and
        their expanded neighborhoods (left_neighborhood, right_neighborhood). A
        value of -1 indicates a boundary case where the region extends to the
        beginning or end of the data.
    """
    n_neighbor = int(sampling_rate * neighborhood_size)
    edges_indices = _extrapolate_edges(
        marker=marker,
        confidence_threshold=confidence_threshold,
        sampling_rate=sampling_rate,
    )
    marker_length = len(marker)

    expanded = edges_indices.with_columns(
        [
            (pl.col("start") - n_neighbor).alias("left_neighborhood"),
            (pl.col("end") + n_neighbor).alias("right_neighborhood"),
        ]
    )

    neighborhoods = expanded.with_columns(
        [
            pl.when(pl.col("start") == -1)
            .then(-1)
            .otherwise(pl.col("left_neighborhood").clip(0, marker_length - 1))
            .alias("left_neighborhood"),
            pl.when(pl.col("end") == -1)
            .then(-1)
            .otherwise(pl.col("right_neighborhood").clip(0, marker_length - 1))
            .alias("right_neighborhood"),
        ]
    )

    return neighborhoods


def _extrapolate_edges(
    marker: np.ndarray, confidence_threshold: float = 0.5, sampling_rate: int = 100
) -> pl.DataFrame:
    """Find edges of maxed out regions and identify them as hills(+) or valleys (-).

    Args:
        marker: Array containing values that mark the probability that each sample is
            maxed-out. Values close to ±1 indicate high likelihood that the value is
            near the limit of the dynamic range, with the sign indicating if it's near
            the upper bound, or lower bound. Values close to zero indicate low
            likelihood of being near range limit.
        confidence_threshold: Threshold for what constitutes a significant change in
            marker values.
        sampling_rate: Sampling rate of acceleration data, typically 100hz following
            interpolation.

    Returns:
        DataFrame containing indices for hills and valleys.
    """
    marker_diff_left = np.concatenate(([0], np.diff(marker)))
    marker_diff_right = np.concatenate((np.diff(marker), [0]))

    out_of_range_threshold = sampling_rate * 5

    positive_left_end = np.where(
        (marker_diff_left > confidence_threshold) & (marker > 0)
    )[0]
    positive_right_start = np.where(
        (marker_diff_right < -confidence_threshold) & (marker > 0)
    )[0]
    hills_df = _align_edges(
        marker_length=len(marker),
        left=positive_left_end,
        right=positive_right_start,
        out_of_range_threshold=out_of_range_threshold,
        sign="hill",
    )

    negative_left_end = np.where(
        (marker_diff_left < -confidence_threshold) & (marker < 0)
    )[0]
    negative_right_start = np.where(
        (marker_diff_right > confidence_threshold) & (marker < 0)
    )[0]
    valleys_df = _align_edges(
        marker_length=len(marker),
        left=negative_left_end,
        right=negative_right_start,
        out_of_range_threshold=out_of_range_threshold,
        sign="valley",
    )

    return pl.concat([hills_df, valleys_df], how="vertical")


def _align_edges(
    marker_length: int,
    left: np.ndarray,
    right: np.ndarray,
    out_of_range_threshold: int,
    sign: Literal["hill", "valley"],
) -> pl.DataFrame:
    """Aligns left and right edges of maxed-out regions and handles edge cases.

    Args:
        marker_length: Length of marker array.
        left: 1D vector of indices indicating the left edge of a potential maxed-out
            region.
        right: 1D vector of indices indicating the right edge of a potential maxed-out
            region.
        out_of_range_threshold: The number of samples that determines if an un matched
            edge extends beyond the data range, or if the edge is spurious. Typically 5
            seconds worth of data.
        sign: Indicates if maxed out region is a hill or valley.

    Returns:
        A DataFrame with the left and right edges of each maxed-out region.

    Raises:
        ValueError: If the difference in length between the left and right arrays is
        greater than 1.
    """
    if np.abs(len(left) - len(right)) >= 2:
        raise ValueError(
            f"Mismatch in {sign} edges. # left: {len(left)}, # right: {len(right)}"
        )

    if len(left) - len(right) == 1:
        if marker_length - left[-1] > out_of_range_threshold:
            right = np.concatenate((right, [-1]))
        elif len(left) == 1:
            left = np.array([])
        else:
            left = left[:-1]

    elif len(left) - len(right) == -1:
        if right[0] > out_of_range_threshold:
            left = np.concatenate(([-1], left))
        elif len(right) == 1:
            right = np.array([])
        else:
            right = right[1:]

    valid_indices = (left != -1) & (right != -1)
    if np.any((right[valid_indices] - left[valid_indices]) < 0) and (len(right) > 1):
        left = left[:-1]
        right = right[1:]

    maxed_out_areas = pl.DataFrame({"start": left, "end": right})

    return maxed_out_areas


def _extrapolate_fit(
    axis: np.ndarray,
    time_numeric: np.ndarray,
    marker: np.ndarray,
    neighbors: pl.DataFrame,
    smoothing: float = 0.6,
    neighborhood_size: float = 0.05,
    sampling_rate: int = 100,
) -> List[Tuple[float, float]]:
    """Extrapolate peak of maxed regions by fitting weighted splines on nearby points.

    Args:
        axis: Original acceleration data along one axis.
        time_numeric: Time series data given in epoch time (ns).
        marker: Array the same size as axis, containing values that mark the
            probability that each sample is maxed-out. Values close to ±1 indicate
            high likelihood that the value is near the limit of the dynamic range, with
            the sign indicating if it's near the upper bound, or lower bound. Values
            close to zero indicate low likelihood of being near range limit.
        neighbors: A polars DataFrame containing the start and end indices for the
            maxed out regions, as well as the neighborhoods to the left and right of
            regions.
        smoothing: Smoothing value for UniveriateSpline. Determines how closely the
            spline adheres to the data points. Higher values will give a smoother
            result.
        neighborhood_size: Duration of neighborhood in seconds to use around each
            maxed-out region for regression fitting. Default is 0.05 (50ms).
        sampling_rate: Sampling rate, default is 100Hz

    Returns:
        A List of Tuples representing the peaks of maxed out regions as
        (timestamp, value) pairs. Times are given in epoch time (ns).
    """
    extrapolate_points = []

    for row in neighbors.iter_rows(named=True):
        left_start, left_end = row["left_neighborhood"], row["start"]
        right_start, right_end = row["end"], row["right_neighborhood"]

        fitted_left = _fit_weighted(
            axis=axis,
            time_numeric=time_numeric,
            marker=marker,
            start=left_start,
            end=left_end,
            smoothing=smoothing,
            neighborhood_size=neighborhood_size,
            sampling_rate=sampling_rate,
        )
        fitted_right = _fit_weighted(
            axis=axis,
            time_numeric=time_numeric,
            marker=marker,
            start=right_start,
            end=right_end,
            smoothing=smoothing,
            neighborhood_size=neighborhood_size,
            sampling_rate=sampling_rate,
        )

        if fitted_left is None or fitted_right is None:
            continue

        middle_t = (time_numeric[left_end] + time_numeric[right_start]) / 2

        left_extrapolated = fitted_left(middle_t)
        right_extrapolated = fitted_right(middle_t)

        extrapolated_value = (
            np.array(left_extrapolated) + np.array(right_extrapolated)
        ) / 2

        extrapolate_points.append((middle_t, extrapolated_value))

    return extrapolate_points


def _fit_weighted(
    axis: np.ndarray,
    time_numeric: np.ndarray,
    marker: np.ndarray,
    start: int,
    end: int,
    smoothing: float = 0.6,
    neighborhood_size: float = 0.05,
    sampling_rate: int = 100,
) -> Optional[interpolate.UnivariateSpline]:
    """Fit weighted spline regression model to a section of the data.

    Args:
        axis: Original acceleration data along one axis.
        time_numeric: Time series data given in epoch time (ns).
        marker: Array the same size as axis, containing values that mark the
            probability that each sample is maxed-out. Values close to ±1 indicate
            high likelihood that the value is near the limit of the dynamic range, with
            the sign indicating if it's near the upper bound, or lower bound. Values
            close to zero indicate low likelihood of being near range limit.
        start: Index marking the beginning of the maxed out zone.
        end: Index marking end of maxed out zone.
        smoothing: Smoothing value for UniveriateSpline. Determines how closely the
            spline adheres to the data points. Higher values will give a smoother
            result. Default is 0.6 as optimized in the MIMS-unit paper.
        neighborhood_size: Duration of neighborhood in seconds to use around each
            maxed-out region for regression fitting. Default is 0.05 (50ms).
        sampling_rate: Sampling rate, default is 100Hz

    Returns:
        A UnivariateSpline function fitted to the data segment with weights based on
        the probability of not being maxed-out. Returns None if the input data is
        insufficient or invalid (-1).
    """
    n_over = int(sampling_rate * neighborhood_size)
    if n_over < 4:
        return None
    if start < 0 or end < 0:
        return None

    sub_t = time_numeric[start : end + 1]
    sub_value = axis[start : end + 1]
    weight = 1 - marker[start : end + 1]

    if len(sub_t) < 2:
        return None

    over_t = np.linspace(sub_t[0], sub_t[-1], n_over)
    over_value = np.interp(over_t, sub_t, sub_value)
    over_weight = np.interp(over_t, sub_t, weight)

    fitted = interpolate.UnivariateSpline(
        over_t, over_value, w=over_weight, s=smoothing
    )

    return fitted


def _extrapolate_interpolate(
    axis: np.ndarray,
    time_numeric: np.ndarray,
    marker: np.ndarray,
    points: list,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Interpolate the original signal with extrapolated points.

    Args:
        axis: Original acceleration data along one axis.
        time_numeric: Time series data given in epoch time (ns).
        marker: Array the same size as axis, containing values that mark the
            probability that each sample is maxed-out. Values close to ±1 indicate
            high likelihood that the value is near the limit of the dynamic range, with
            the sign indicating if it's near the upper bound, or lower bound. Values
            close to zero indicate low likelihood of being near range limit.
        points: List of tuples with extrapolated points (time, value). Times are given
            in epoch time (ns).
        confidence_threshold: Threshold for maxed-out identification.

    Returns:
        np.ndarray of axis data, with extrapolated values.
    """
    non_maxed_out_mask = np.abs(marker) < confidence_threshold
    num_non_maxed = np.sum(non_maxed_out_mask)

    if num_non_maxed / len(marker) < 0.3:
        times = time_numeric
        values = axis
    else:
        times = time_numeric[non_maxed_out_mask]
        values = axis[non_maxed_out_mask]

        if points and len(points) > 0:
            points_time = np.array([point[0] for point in points])
            points_value = np.array([point[1] for point in points])

            times = np.concatenate([times, points_time])
            values = np.concatenate([values, points_value])

            sort_idx = np.argsort(times)
            times = times[sort_idx]
            values = values[sort_idx]

    interp_function = interpolate.CubicSpline(times, values, bc_type="natural")
    interp_values = interp_function(time_numeric)

    return interp_values


def butterworth_filter(
    acceleration: models.Measurement,
    sampling_rate: int = 100,
    cutoffs: tuple[float, float] = (0.2, 5.0),
    order: int = 4,
) -> models.Measurement:
    """Apply butterworth IIR filter to acceleration data.

    Implements third portion of MIMS algorithm following interpolate, and extrapolation.

    Args:
        acceleration: Acceleration data to be filtered.
        sampling_rate: Sampling rate of acceleration data in Hz.
        cutoffs: Cutoff values for bandpass filter.
        order: Order of the filter, defaults to 4th order.


    Returns:
        Acceleration Measurement of filtered data.
    """
    normalized_cutoffs = [cutoff / (sampling_rate * 0.5) for cutoff in cutoffs]
    b, a = signal.butter(N=order, Wn=normalized_cutoffs, btype="bandpass")

    filtered_data = [
        signal.lfilter(b=b, a=a, x=column) for column in acceleration.measurements.T
    ]

    return models.Measurement(
        measurements=np.column_stack(filtered_data), time=acceleration.time
    )


def aggregate_mims(
    acceleration: models.Measurement,
    epoch: float = 60.0,
    sampling_rate: int = 100,
    *,
    rectify: bool = True,
    truncate: bool = True,
) -> models.Measurement:
    """Calculate the area under the curve(AUC), per epoch, per axis.

    When an epoch has less than 90% of the expected values (based on the sampling rate
    and epoch length), the AUC for that epoch is given as -1 for each axis. If rectify
    is True, any axis with values below -150 will have the AUC value for that axis
    be -1 for that epoch. Finally, following integration, any value greater than (16 *
    sampling_rate * epoch) will be set to -1.

    Args:
        acceleration: Acceleration data to be aggregated.
        epoch: The desired epoch length in seconds that data will be aggregated over.
        sampling_rate: The sampling rate of the accelerometer data in Hz.
        rectify: Specifies if data should be rectified before integration. If True any
            value below -150 will assign the value of that axis to -1 for that epoch.
            Additionally the absolute value of accelerometer data will be used for
            integration.
        truncate: Specifies if data <= 0.001 should be truncated to 0.

    Returns:
        A models.Measurement instance with the area under the curve values for each
        epoch.
    """
    if epoch <= 0:
        raise ValueError("Epoch length must be greater than 0.")

    epoch_ns = int(epoch * 1e9)
    acceleration_df = pl.DataFrame(
        {
            "time": acceleration.time,
            "x": acceleration.measurements[:, 0],
            "y": acceleration.measurements[:, 1],
            "z": acceleration.measurements[:, 2],
        }
    ).set_sorted("time")

    result = acceleration_df.group_by_dynamic(
        "time",
        every=f"{epoch_ns}ns",
    ).map_groups(
        lambda group: _aggregate_epoch(
            group=group,
            epoch=epoch,
            sampling_rate=sampling_rate,
            rectify=rectify,
        ),
        schema={
            "time": pl.Datetime("ns"),
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
        },
    )

    result = result.with_columns(pl.col("time").dt.truncate(f"{int(epoch)}s"))

    aggregated_measure = models.Measurement(
        measurements=result.select(["x", "y", "z"]).to_numpy(),
        time=result["time"].cast(pl.Datetime("ns")),
    )

    truncate_threshold = 1e-4 * epoch * sampling_rate
    if truncate:
        aggregated_measure.measurements = np.where(
            aggregated_measure.measurements <= truncate_threshold,
            0,
            aggregated_measure.measurements,
        )

    return aggregated_measure


def _aggregate_epoch(
    group: pl.DataFrame,
    epoch: float = 60.0,
    sampling_rate: int = 100,
    *,
    rectify: bool = True,
) -> pl.DataFrame:
    """Calculate the area under the curve(AUC), per epoch.

    Args:
        group: The epoch given by .map_groups()
        epoch: The desired epoch length in seconds that data will be aggregated over.
            Defaults to 1 minute as per MIMS-unit paper.
        sampling_rate: The sampling rate of the accelerometer data in Hz.
        rectify: Specifies if data should be rectified before integration. If True any
            value below -150 will assign the value of that axis to -1 for that epoch.
            Additionally the absolute value of accelerometer data will be used for
            integration.

    Returns:
        A polars DataFrame containing the XYZ AUC values for one epoch.
    """
    timestamps = group["time"].cast(pl.Int64).to_numpy()

    if len(timestamps) < 0.9 * sampling_rate * epoch:
        logger.debug("Not enough data to calculate AUC")
        return pl.DataFrame(
            {"time": [group["time"].min()], "x": [-1.0], "y": [-1.0], "z": [-1.0]}
        )

    values = group.select(["x", "y", "z"]).to_numpy()

    times_sec = timestamps / 1e9

    if rectify:
        area = integrate.trapezoid(y=np.abs(values), x=times_sec, axis=0)
        low_values = np.any(values <= -150, axis=0)
        if np.any(low_values):
            logger.warning("Values below -150 detected. Consider checking your data.")
        area = np.where(low_values, -1, area)
    else:
        area = integrate.trapezoid(y=values, x=times_sec, axis=0)

    max_value = 16 * sampling_rate * epoch
    area = np.where(np.logical_or(area >= max_value, area < 0), -1.0, area)

    return pl.DataFrame(
        {"time": [group["time"].min()], "x": [area[0]], "y": [area[1]], "z": [area[2]]}
    )


def combine_mims(
    acceleration: models.Measurement,
    combination_method: Literal["sum", "vector_magnitude"] = "sum",
) -> models.Measurement:
    """Combine MIMS values of xyz axis into one MIMS value.

    If any value in an epoch is -1, the 'combined_mims' value for that epoch will
    be set to -1 to flag it as an invalid epoch.

    Args:
        acceleration: An object containing per-axis MIMS measurements and their
            corresponding timestamps.
        combination_method: The method to combine MIMS values across axes. Defaults to
            "sum".

    Returns:
            A Measurement object with combined MIMS values and corresponding timestamps.
    """
    row_contains_negative = np.any(acceleration.measurements == -1, axis=1)

    if combination_method == "sum":
        row_sum = np.sum(acceleration.measurements, axis=1)
        combined_mims = np.where(row_contains_negative, -1, row_sum)
    elif combination_method == "vector_magnitude":
        row_norm = np.linalg.norm(acceleration.measurements, axis=1)
        combined_mims = np.where(row_contains_negative, -1, row_norm)
    else:
        raise ValueError(
            f"Invalid combination_method given:{combination_method}."
            "Must be 'sum' or 'vector_magnitude'. "
        )

    return models.Measurement(measurements=combined_mims, time=acceleration.time)
