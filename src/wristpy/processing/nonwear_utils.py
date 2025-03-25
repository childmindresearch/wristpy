"""This module contains helper functions for aggregated nonwear detection outputs."""

import datetime

import numpy as np
import polars as pl

from wristpy.core import computations, models


def majority_vote_non_wear(
    *nonwear_measurements: models.Measurement,
    temporal_resolution: float = 60.0,
) -> models.Measurement:
    """This function applies a majority vote on any number of nonwear Measurements.

    The _time_fix function is used to ensure that all nonwear Measurements have the
    same start and endpoints. Then each nonwear Measurement is resampled to the same
    temporal_resoltion. A majority vote is taken at each time point to determine the
    new nonwear Measurement.
    In case of an even number of inputs, the majority is rounded up.

    Args:
        *nonwear_measurements: Variable number of nonwear algorithm outputs.
        temporal_resolution: The temporal resolution of the output, in seconds.
            Defaults to 60.0.

    Returns:
        A new Measurement instance with the combined nonwear detection,
        at a new temporal resolution.
    """
    num_measurements = len(nonwear_measurements)
    if num_measurements % 2 == 0:
        majority_threshold = num_measurements // 2 + 1
    else:
        majority_threshold = int(np.ceil(num_measurements / 2))

    min_start_time = min(measurement.time[0] for measurement in nonwear_measurements)
    max_end_time = max(measurement.time[-1] for measurement in nonwear_measurements)

    measurement_sum = None
    for measurement in nonwear_measurements:
        time_adjust_measurement = _time_fix(measurement, max_end_time, min_start_time)
        resampled_measurement = computations.resample(
            time_adjust_measurement, temporal_resolution
        )

        binary_nonwear = np.where(resampled_measurement.measurements >= 0.5, 1, 0)

        if measurement_sum is None:
            measurement_sum = binary_nonwear
            reference_time = resampled_measurement.time
        else:
            measurement_sum += binary_nonwear

    nonwear_value = np.where(measurement_sum >= majority_threshold, 1, 0)  # type: ignore[operator] #measurement_sum is never None type

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
