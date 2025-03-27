"""Test the function of the nonwear_utils module."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from wristpy.core import models
from wristpy.processing import nonwear_utils


def test_nonwear_majority_vote() -> None:
    """Tests the majority vote function for nonwear."""
    time1 = [
        datetime(1990, 1, 1, 1, 1) + timedelta(seconds=60 * idx) for idx in range(900)
    ]
    time3 = [
        datetime(1990, 1, 1, 1, 1, 10) + timedelta(seconds=100 * idx)
        for idx in range(11)
    ]
    nonwear1 = models.Measurement(
        measurements=np.ones(len(time1)),
        time=pl.Series("time", time1, dtype=pl.Datetime("ns")),
    )
    nonwear2 = models.Measurement(
        measurements=np.ones(len(time1)),
        time=pl.Series("time", time1, dtype=pl.Datetime("ns")),
    )
    nonwear3 = models.Measurement(
        measurements=np.ones(len(time3)),
        time=pl.Series("time", time3, dtype=pl.Datetime("ns")),
    )

    nonwear = nonwear_utils.majority_vote_non_wear(
        [nonwear1, nonwear2, nonwear3],
        temporal_resolution=5,
    )

    assert np.all(nonwear.measurements == 1)


def test_nonwear_majority_vote_even() -> None:
    """Tests the majority vote function for nonwear, with even input."""
    time1 = [
        datetime(1990, 1, 1, 1, 1) + timedelta(seconds=60 * idx) for idx in range(900)
    ]
    nonwear1 = models.Measurement(
        measurements=np.ones(len(time1)),
        time=pl.Series("time", time1, dtype=pl.Datetime("ns")),
    )
    nonwear2 = models.Measurement(
        measurements=np.zeros(len(time1)),
        time=pl.Series("time", time1, dtype=pl.Datetime("ns")),
    )

    nonwear = nonwear_utils.majority_vote_non_wear(
        [nonwear1, nonwear2],
        temporal_resolution=5,
    )

    assert np.all(nonwear.measurements == 0)


def test_nonwear_majority_vote_empty() -> None:
    """Tests the majority vote function for nonwear when no inputs provided."""
    with pytest.raises(ValueError):
        nonwear_utils.majority_vote_non_wear(
            nonwear_measurements=[],
            temporal_resolution=5,
        )


def test_nonwear_dispatcher_default() -> None:
    """Tests the nonwear dispatcher function."""
    num_samples = int(1e5)
    time_list = [
        datetime(2024, 5, 2) + timedelta(milliseconds=100 * i)
        for i in range(num_samples)
    ]
    time = pl.Series("time", time_list, dtype=pl.Datetime("ns"))
    acceleration = models.Measurement(measurements=np.ones((num_samples, 3)), time=time)
    temperature = models.Measurement(
        measurements=np.random.randint(low=22, high=32, size=num_samples), time=time
    )

    nonwear_array = nonwear_utils.get_nonwear_measurements(acceleration, temperature)

    assert isinstance(nonwear_array, models.Measurement)


def test_nonwear_dispatcher_multiple() -> None:
    """Tests nonwear dispatcher with multiple nonwear alogirhtms requested."""
    num_samples = int(1e5)
    time_list = [
        datetime(2024, 5, 2) + timedelta(milliseconds=100 * i)
        for i in range(num_samples)
    ]
    time = pl.Series("time", time_list, dtype=pl.Datetime("ns"))
    acceleration = models.Measurement(measurements=np.ones((num_samples, 3)), time=time)
    temperature = models.Measurement(
        measurements=np.random.randint(low=22, high=32, size=num_samples), time=time
    )

    nonwear_array = nonwear_utils.get_nonwear_measurements(
        acceleration, temperature, ["cta", "detach", "ggir"]
    )

    assert isinstance(nonwear_array, models.Measurement)


def test_nonwear_dispatcher_unknown_algo() -> None:
    """Tests nonwear dispatcher with incorrect algorithm."""
    num_samples = int(1e5)
    time_list = [
        datetime(2024, 5, 2) + timedelta(milliseconds=100 * i)
        for i in range(num_samples)
    ]
    time = pl.Series("time", time_list, dtype=pl.Datetime("ns"))
    acceleration = models.Measurement(measurements=np.ones((num_samples, 3)), time=time)
    temperature = models.Measurement(
        measurements=np.random.randint(low=22, high=32, size=num_samples), time=time
    )

    with pytest.raises(ValueError):
        nonwear_utils.get_nonwear_measurements(acceleration, temperature, ["unknown"])  # type: ignore [list-item] #Violating Literal to raise ValueError
