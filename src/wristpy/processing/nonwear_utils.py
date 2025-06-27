"""This module contains helper functions for aggregated nonwear detection outputs."""

import datetime
from typing import Literal, Sequence, Union

import numpy as np
import polars as pl

from wristpy.core import computations, models
from wristpy.processing import metrics


def _majority_vote_non_wear(
    nonwear_measurements: Sequence[models.Measurement],
    temporal_resolution: float = 60.0,
) -> models.Measurement:
    """This function applies a majority vote on any number of nonwear Measurements.

    The _time_fix function is used to ensure that all nonwear Measurements have the
    same start and endpoints. Then each nonwear Measurement is resampled to the same
    temporal_resolution. A majority vote is taken at each time point to determine the
    new nonwear Measurement.
    In case of an even number of inputs, the majority is rounded up.
    If only one nonwear Measurement is provided, it is returned as is.

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

    if num_measurements == 1:
        return nonwear_measurements[0]

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
    temperature: Union[models.Measurement, None] = None,
    non_wear_algorithms: Sequence[Literal["ggir", "cta", "detach"]] = ["ggir"],
) -> models.Measurement:
    """Non-wear measurement dispatcher.

    This function chooses which non-wear detection algorithm(s) to use based on the
    provided sequence of valid algorithm names.

    Args:
        calibrated_acceleration: The calibrated acceleration data
        temperature: Temperature data if available
        non_wear_algorithms: One or more algorithm names to use.


    Returns:
        A non-wear Measurement object.

    Raises:
        ValueError:
            If the CTA or DETACH algorithm is requested without temperature data.
            If an unknown algorithm is specified.

    """
    temp_dependent_algorithm = {"cta", "detach"}
    if (
        any(algorithm in temp_dependent_algorithm for algorithm in non_wear_algorithms)
        and temperature is None
    ):
        raise ValueError(
            "Temperature data is required for the CTA and DETACH nonwear algorithms."
        )

    non_wear_algorithm_functions = {
        "ggir": lambda: metrics.detect_nonwear(calibrated_acceleration),
        "cta": lambda: metrics.combined_temp_accel_detect_nonwear(
            acceleration=calibrated_acceleration,
            temperature=temperature,  # type: ignore[arg-type] #protected by if statement
        ),
        "detach": lambda: metrics.detach_nonwear(
            acceleration=calibrated_acceleration,
            temperature=temperature,  # type: ignore[arg-type] #protected by if statement
        ),
    }

    if any(
        algorithm not in non_wear_algorithm_functions
        for algorithm in non_wear_algorithms
    ):
        raise ValueError("An unknown algorithm was specified.")

    results = [
        non_wear_algorithm_functions[algorithm]() for algorithm in non_wear_algorithms
    ]

    if len(results) > 1:
        return _majority_vote_non_wear(results)
    return results[0]


def synchronize_measurements(
    data_measurement: models.Measurement,
    reference_measurement: models.Measurement,
    epoch_length: float = 5.0,
) -> models.Measurement:
    """This function is used to match a Measurement object to a reference Measurement.

    This function ensures that a Measurement object and reference Measurement times
    are synced up. This is accomplished by first resampling a Measurement object to
    the specified temporal resolution.
    It also ensures that a Measurement object is a binary array, where 1 indicates
    nonwear and 0 indicates wear.
    It then truncates a Measurement object to match the reference
    Measurement time points.

    Args:
        data_measurement: The nonwear array to clean up.
        reference_measurement: The reference measurement to use for resampling.
        epoch_length: The temporal resolution of the output, in seconds.
            Defaults to 5.0.

    Returns:
        A new Measurement instance with the cleaned up nonwear detection,
        returned as a boolean (True == nonwear).
    """
    time_fix_nonwear = _time_fix(
        data_measurement, reference_measurement.time[-1], reference_measurement.time[0]
    )
    resampled_nonwear = computations.resample(time_fix_nonwear, epoch_length)
    binary_nonwear = np.where(resampled_nonwear.measurements >= 0.5, 1, 0)

    ref_df = pl.DataFrame({"time": reference_measurement.time})

    nonwear_df = pl.DataFrame(
        {
            "time": resampled_nonwear.time,
            "idx": pl.arange(0, len(resampled_nonwear.time), eager=True),
        }
    )

    joined = ref_df.join(nonwear_df, on="time", how="inner")

    matched_indices = joined["idx"].to_numpy()
    matched_values = binary_nonwear[matched_indices]

    return models.Measurement(
        measurements=matched_values.astype(bool), time=joined["time"]
    )
