"""This module contains helper functions for aggregated nonwear detection outputs."""

import datetime

import numpy as np
import polars as pl

from wristpy.core import computations, models


def majority_vote_non_wear(
    nonwear_ggir: models.Measurement,
    nonwear_cta: models.Measurement,
    nonwear_detach: models.Measurement,
    temporal_resolution: float = 60.0,
) -> models.Measurement:
    """This function applies a majority vote on the three possible nonwear outputs.

    The three algorithms are resampled to the same sampling rate, and a majority vote
    is taken at each time point to determine the new nonwear time.


    This assumes the nonwear measurements have the same ending time stamp.

    Args:
        nonwear_ggir: The nonwear algorithm output from the GGIR algorithm.
        nonwear_cta: The nonwear algorithm output from the CTA algorithm.
        nonwear_detach: The nonwear algorithm output from the detach algorithm.
        temporal_resolution: The temporal resolution of the output, in seconds.
            Defaults to 60.0.

    Returns:
        A new Measurement instance with the combined nonwear detection,
        at a new temporal resolution.
    """
    min_start_time = min(
        [nonwear_ggir.time[0], nonwear_cta.time[0], nonwear_detach.time[0]]
    )

    max_end_time = max(
        [nonwear_ggir.time[-1], nonwear_cta.time[-1], nonwear_detach.time[-1]]
    )

    nonwear_ggir = _time_fix(nonwear_ggir, max_end_time, min_start_time)
    nonwear_cta = _time_fix(nonwear_cta, max_end_time, min_start_time)
    nonwear_detach = _time_fix(nonwear_detach, max_end_time, min_start_time)

    nonwear_ggir = computations.resample(nonwear_ggir, temporal_resolution)
    nonwear_cta = computations.resample(nonwear_cta, temporal_resolution)
    nonwear_detach = computations.resample(nonwear_detach, temporal_resolution)

    nonwear_ggir.measurements = np.where(nonwear_ggir.measurements >= 0.5, 1, 0)
    nonwear_cta.measurements = np.where(nonwear_cta.measurements >= 0.5, 1, 0)

    nonwear_value = np.where(
        (
            nonwear_ggir.measurements
            + nonwear_cta.measurements
            + nonwear_detach.measurements
        )
        >= 2,
        1,
        0,
    )

    return models.Measurement(measurements=nonwear_value, time=nonwear_ggir.time)


def combined_ggir_detach_nonwear(
    nonwear_ggir: models.Measurement,
    nonwear_detach: models.Measurement,
    temporal_resolution: float = 60.0,
) -> models.Measurement:
    """This function combines the GGIR and DETACH nonwear detection outputs.

    The two algorithms are resampled to the same sampling rate, and when both
    algorithms agree on nonwear, the new nonwear outputs is set to 1.

    This assumes the nonwear measurements have the same ending time stamp.

    Args:
        nonwear_ggir: The nonwear algorithm output from the GGIR algorithm.
        nonwear_detach: The nonwear algorithm output from the DETACH algorithm.
        temporal_resolution: The temporal resolution of the output, in seconds.
            Defaults to 60.0.

    Returns:
        A new Measurement instance at a new temporal resolution.
    """
    min_start_time = min([nonwear_ggir.time[0], nonwear_detach.time[0]])
    max_end_time = max([nonwear_ggir.time[-1], nonwear_detach.time[-1]])

    nonwear_ggir = _time_fix(nonwear_ggir, max_end_time, min_start_time)
    nonwear_detach = _time_fix(nonwear_detach, max_end_time, min_start_time)

    nonwear_ggir = computations.resample(nonwear_ggir, temporal_resolution)
    nonwear_detach = computations.resample(nonwear_detach, temporal_resolution)

    nonwear_ggir.measurements = np.where(nonwear_ggir.measurements >= 0.5, 1, 0)

    nonwear_value = np.where(
        (nonwear_ggir.measurements + nonwear_detach.measurements) == 2,
        1,
        0,
    )

    return models.Measurement(measurements=nonwear_value, time=nonwear_ggir.time)


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
