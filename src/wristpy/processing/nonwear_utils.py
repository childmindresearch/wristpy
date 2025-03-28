"""This module contains helper functions for aggregated nonwear detection outputs."""

import datetime
from typing import Callable, Dict, Literal, Sequence

import numpy as np
import polars as pl

from wristpy.core import computations, models
from wristpy.processing import metrics


def majority_vote_non_wear(
    nonwear_measurements: Sequence[models.Measurement],
    temporal_resolution: float = 60.0,
) -> models.Measurement:
    """This function applies a majority vote on any number of nonwear Measurements.

    The _time_fix function is used to ensure that all nonwear Measurements have the
    same start and endpoints. Then each nonwear Measurement is resampled to the same
    temporal_resoltion. A majority vote is taken at each time point to determine the
    new nonwear Measurement.
    In case of an even number of inputs, the majority is rounded up.

    Args:
        nonwear_measurements: A sequence (ex. List, Tuple, ...) of Measurement objects.
        temporal_resolution: The temporal resolution of the output, in seconds.
            Defaults to 60.0.

    Returns:
        A new Measurement instance with the combined nonwear detection,
        at a new temporal resolution.

    Raises:
        ValueError: If the number of nonwear measurements is 0.
    """
    num_measurements = len(nonwear_measurements)

    if num_measurements == 0:
        raise ValueError("At least one nonwear measurement is required.")

    majority_threshold = num_measurements / 2

    min_start_time = min(measurement.time[0] for measurement in nonwear_measurements)
    max_end_time = max(measurement.time[-1] for measurement in nonwear_measurements)

    first_resampled_measurement = computations.resample(
        _time_fix(nonwear_measurements[0], max_end_time, min_start_time),
        temporal_resolution,
    )
    measurement_sum = np.zeros_like(first_resampled_measurement.measurements)
    reference_time = first_resampled_measurement.time

    for measurement in nonwear_measurements:
        time_adjust_measurement = _time_fix(measurement, max_end_time, min_start_time)
        resampled_measurement = computations.resample(
            time_adjust_measurement, temporal_resolution
        )

        binary_nonwear = np.where(resampled_measurement.measurements >= 0.5, 1, 0)
        measurement_sum += binary_nonwear

    nonwear_value = np.where(measurement_sum > majority_threshold, 1, 0)

    return models.Measurement(measurements=nonwear_value, time=reference_time)


def _time_fix(
    nonwear: models.Measurement,
    end_time: datetime.datetime,
    start_time: datetime.datetime,
) -> models.Measurement:
    """Helper function to fix the time of the nonwear measurements.

    This function appends start/end points to the nonwear measurements based on
    previously computed reference start and end points.

    Args:
        nonwear: The nonwear measurement to adjust start/end time points.
        end_time: The maximum end time of the nonwear measurements.
        start_time: The minimum start time of the nonwear measurements.

    Returns:
        The nonwear measurement with the time fixed.
    """
    if nonwear.time[0] > start_time:
        nonwear.time = pl.concat(
            [pl.Series([start_time], dtype=pl.Datetime("ns")), nonwear.time]
        )
        nonwear.measurements = np.append(nonwear.measurements[0], nonwear.measurements)

    if nonwear.time[-1] < end_time:
        nonwear.time.append(pl.Series([end_time], dtype=pl.Datetime("ns")))
        nonwear.measurements = np.append(nonwear.measurements, nonwear.measurements[-1])
    return nonwear


def get_nonwear_measurements(
    calibrated_acceleration: models.Measurement,
    temperature: models.Measurement,
    non_wear_algorithms: Sequence[Literal["ggir", "cta", "detach"]] = ["ggir"],
) -> models.Measurement:
    """Non-wear measurement dispatcher.

    This function chooses which non-wear detection algorithm(s) to use based on the
    provided sequence of valid algorithm names.

    Args:
        calibrated_acceleration: The calibrated acceleration data
        temperature: Temperature data if available
        algorithms: One or more algorithm names to use

    Returns:
        A non-wear Measurement object.

    Raises:
        ValueError: If an unknown algorithm is specified.
    """
    non_wear_algorithm_functions: Dict[str, Callable] = {
        "ggir": lambda: metrics.detect_nonwear(calibrated_acceleration),
        "cta": lambda: metrics.combined_temp_accel_detect_nonwear(
            acceleration=calibrated_acceleration, temperature=temperature
        ),
        "detach": lambda: metrics.detach_nonwear(
            acceleration=calibrated_acceleration, temperature=temperature
        ),
    }

    results = []
    for algorithm in non_wear_algorithms:
        if algorithm not in non_wear_algorithm_functions:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        results.append(non_wear_algorithm_functions[algorithm]())

    if len(results) > 1:
        return majority_vote_non_wear(results)
    else:
        return results[0]
