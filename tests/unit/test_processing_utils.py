"""Test the function of the nonwear_utils module."""

from datetime import datetime, timedelta

import numpy as np
import numpy.testing as npt
import polars as pl
import pytest

from wristpy.core import models
from wristpy.processing import processing_utils


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

    nonwear = processing_utils._majority_vote_non_wear(
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

    nonwear = processing_utils._majority_vote_non_wear(
        [nonwear1, nonwear2],
        temporal_resolution=5,
    )

    assert np.all(nonwear.measurements == 0)


def test_nonwear_majority_vote_empty() -> None:
    """Tests the majority vote function for nonwear when no inputs provided."""
    with pytest.raises(ValueError):
        processing_utils._majority_vote_non_wear(
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

    nonwear_array = processing_utils.get_nonwear_measurements(acceleration, temperature)

    assert isinstance(nonwear_array, models.Measurement)


def test_nonwear_dispatcher_multiple() -> None:
    """Tests nonwear dispatcher with multiple nonwear algorithms requested."""
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

    nonwear_array = processing_utils.get_nonwear_measurements(
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
        processing_utils.get_nonwear_measurements(
            acceleration,
            temperature,
            ["unknown"],  # type: ignore [list-item] #Violating Literal to raise ValueError
        )


def test_nonwear_dispatcher_missing_temperature() -> None:
    """Tests nonwear dispatcher with missing temperature."""
    num_samples = int(1e5)
    time_list = [
        datetime(2024, 5, 2) + timedelta(milliseconds=100 * i)
        for i in range(num_samples)
    ]
    time = pl.Series("time", time_list, dtype=pl.Datetime("ns"))
    acceleration = models.Measurement(measurements=np.ones((num_samples, 3)), time=time)

    with pytest.raises(ValueError):
        processing_utils.get_nonwear_measurements(
            acceleration, non_wear_algorithms=["cta"]
        )


def test_nonwear_default_synchronize_measurements() -> None:
    """Tests synchronize function for nonwear measurement.

    Tests synchronize_measurements() function with default parameters
    for nonwear measurement object.
    """
    nonwear_period_len = 15  # in min
    num_periods = 19
    num_samples = int(nonwear_period_len * num_periods * 12)
    data_time_list = [
        datetime(2024, 12, 8) + timedelta(minutes=i * nonwear_period_len)
        for i in range(num_periods)
    ]

    ref_time_list = [
        datetime(2024, 12, 8) + timedelta(seconds=5 * i) for i in range(num_samples)
    ]

    data_time = pl.Series("time", data_time_list, dtype=pl.Datetime("ns"))
    ref_time = pl.Series("time", ref_time_list, dtype=pl.Datetime("ns"))

    poisson_x = [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    poisson_time = range(
        0, len(poisson_x) * nonwear_period_len * 12, nonwear_period_len * 12
    )
    resampled_time = range(0, len(poisson_x) * nonwear_period_len * 12, 1)
    expected_nonwear_epoch = np.interp(resampled_time, poisson_time, poisson_x)
    binary_nonwear_epoch = [1 if item >= 0.5 else 0 for item in expected_nonwear_epoch]

    nonwear_array = models.Measurement(
        measurements=np.asarray(poisson_x), time=data_time
    )
    ref_measurement = models.Measurement(
        measurements=np.ones((num_periods * nonwear_period_len,)), time=ref_time
    )

    nonwear_epoch = processing_utils.synchronize_measurements(
        data_measurement=nonwear_array,
        reference_measurement=ref_measurement,
        epoch_length=5,
    )

    npt.assert_array_equal(nonwear_epoch.measurements, np.asarray(binary_nonwear_epoch))


def test_sib_default_synchronize_sleep_bouts() -> None:
    """Tests synchronize function for sib measurement.

    Tests _synchronize_sleep_bouts() with default parameters for
    sustained inactivity bout.
    """
    nonwear_period_len = 15  # in min
    num_periods = 19
    num_samples = int(nonwear_period_len * num_periods * 12)
    data_time_list = [
        datetime(2024, 12, 8) + timedelta(minutes=i * nonwear_period_len)
        for i in range(num_periods)
    ]

    ref_time_list = [
        datetime(2024, 12, 8) + timedelta(seconds=5 * i) for i in range(num_samples)
    ]

    data_time = pl.Series("time", data_time_list, dtype=pl.Datetime("ns"))
    ref_time = pl.Series("time", ref_time_list, dtype=pl.Datetime("ns"))

    poisson_x = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    expected_sib = [
        bool(item) for item in poisson_x for _ in range(nonwear_period_len * 12)
    ]

    sib_array = models.Measurement(measurements=np.asarray(poisson_x), time=data_time)
    ref_measurement = models.Measurement(
        measurements=np.ones((num_periods * nonwear_period_len,)), time=ref_time
    )

    sleep_parameter_sync = processing_utils._synchronize_sleep_bouts(
        sib_array,
        ref_measurement,
    )

    npt.assert_array_equal(sleep_parameter_sync.measurements, np.asarray(expected_sib))


def test_spt_default_synchronize_sleep_bouts() -> None:
    """Tests synchronize function for spt measurement.

    Tests _synchronize_sleep_bouts() with default parameters for
    sleep period time.
    """
    nonwear_period_len = 5 / 60  # in min
    num_periods = 19
    num_samples = int(num_periods)
    data_time_list = [
        datetime(2024, 12, 8) + timedelta(minutes=i * nonwear_period_len)
        for i in range(num_periods)
    ]

    ref_time_list = [
        datetime(2024, 12, 8) + timedelta(seconds=5 * i) for i in range(num_samples)
    ]

    data_time = pl.Series("time", data_time_list, dtype=pl.Datetime("ns"))
    ref_time = pl.Series("time", ref_time_list, dtype=pl.Datetime("ns"))

    poisson_x = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    expected_sib = [bool(item) for item in poisson_x for _ in range(1)]

    sib_array = models.Measurement(measurements=np.asarray(poisson_x), time=data_time)
    ref_measurement = models.Measurement(
        measurements=np.ones((num_periods,)), time=ref_time
    )

    sleep_parameter_sync = processing_utils._synchronize_sleep_bouts(
        sib_array,
        ref_measurement,
    )

    npt.assert_array_equal(sleep_parameter_sync.measurements, np.asarray(expected_sib))
