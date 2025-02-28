"""Test the function of computations module."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from wristpy.core import computations, models

SIGNAL_LENGTH = 20
EPOCH_LENGTH = 5


def test_moving_mean_epoch_length_is_negative() -> None:
    """Test error if the epoch length is negative."""
    tmp_data = np.array([1, 2, 3])
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + timedelta(seconds=i) for i in range(3)]
    time = pl.Series("time", dummy_datetime_list)
    tmp_measurement = models.Measurement(measurements=tmp_data, time=time)

    with pytest.raises(ValueError):
        computations.moving_mean(tmp_measurement, epoch_length=-1)


def test_moving_mean_one_column() -> None:
    """Test the functionality of the moving mean function for 1D Measurement."""
    test_data = np.arange(0, SIGNAL_LENGTH)
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [
        dummy_date + timedelta(seconds=i) for i in range(SIGNAL_LENGTH)
    ]
    test_time = pl.Series("time", dummy_datetime_list)
    expected_time_shape = math.ceil(test_time.shape[0] / EPOCH_LENGTH)
    expected_mean = np.array([2.0, 7.0, 12.0, 17.0])
    test_measurement = models.Measurement(measurements=test_data, time=test_time)

    test_measurement_mean = computations.moving_mean(
        test_measurement, epoch_length=EPOCH_LENGTH
    )

    assert np.allclose(test_measurement_mean.measurements, expected_mean)
    assert test_measurement_mean.measurements.ndim == expected_mean.ndim
    assert test_measurement_mean.time.shape[0] == expected_time_shape


def test_moving_mean_three_columns() -> None:
    """Test the functionality of the moving mean function for three column array."""
    test_data = np.arange(0, SIGNAL_LENGTH * 3).reshape(SIGNAL_LENGTH, 3)
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [
        dummy_date + timedelta(seconds=i) for i in range(SIGNAL_LENGTH)
    ]
    test_time = pl.Series("time", dummy_datetime_list)
    expected_time_shape = math.ceil(test_time.shape[0] / EPOCH_LENGTH)
    expected_mean = np.array(
        ([[6.0, 7.0, 8.0], [21.0, 22.0, 23.0], [36.0, 37.0, 38.0], [51.0, 52.0, 53.0]])
    )
    test_measurement = models.Measurement(measurements=test_data, time=test_time)

    test_measurement_mean = computations.moving_mean(
        test_measurement, epoch_length=EPOCH_LENGTH
    )

    assert np.allclose(test_measurement_mean.measurements, expected_mean)
    assert test_measurement_mean.measurements.shape[1] == test_data.shape[1]
    assert test_measurement_mean.time.shape[0] == expected_time_shape


def test_moving_std_epoch_length_is_negative() -> None:
    """Test error if the epoch length is negative."""
    tmp_data = np.array([1, 2, 3])
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + timedelta(seconds=i) for i in range(3)]
    time = pl.Series("time", dummy_datetime_list)
    tmp_measurement = models.Measurement(measurements=tmp_data, time=time)

    with pytest.raises(ValueError):
        computations.moving_std(tmp_measurement, epoch_length=-1)


@pytest.mark.parametrize(
    "test_array, expected_std",
    [
        (np.ones(20), np.array([0.0, 0.0, 0.0, 0.0])),
        (
            np.array([-2, -1, 0, 1, 2] * 4),
            np.array([1.581139] * 4),
        ),
    ],
)
def test_moving_std_one_column(
    test_array: np.ndarray, expected_std: np.ndarray
) -> None:
    """Test the functionality of the moving std function for 1D Measurement."""
    SIGNAL_LENGTH = 20
    EPOCH_LENGTH = 5

    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [
        dummy_date + timedelta(seconds=i) for i in range(SIGNAL_LENGTH)
    ]
    test_time = pl.Series("time", dummy_datetime_list)
    expected_time_shape = math.ceil(test_time.shape[0] / EPOCH_LENGTH)
    test_measurement = models.Measurement(measurements=test_array, time=test_time)

    test_measurement_std = computations.moving_std(
        test_measurement, epoch_length=EPOCH_LENGTH
    )

    assert np.allclose(test_measurement_std.measurements, expected_std)
    assert test_measurement_std.measurements.ndim == expected_std.ndim
    assert test_measurement_std.time.shape[0] == expected_time_shape


@pytest.mark.parametrize(
    "test_data, expected_std",
    [
        (np.ones(20), np.array([0.0, 0.0, 0.0, 0.0])),
        (
            np.array([-2, -1, 0, 1, 2] * 4),
            np.array([1.581139] * 4),
        ),
    ],
)
def test_moving_std_three_columns(
    test_data: np.ndarray, expected_std: np.ndarray
) -> None:
    """Test the functionality of the moving std function for three column array."""
    SIGNAL_LENGTH = 20
    EPOCH_LENGTH = 5

    test_data = np.column_stack([test_data] * 3)
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [
        dummy_date + timedelta(seconds=i) for i in range(SIGNAL_LENGTH)
    ]
    test_time = pl.Series("time", dummy_datetime_list)
    expected_time_shape = math.ceil(test_time.shape[0] / EPOCH_LENGTH)
    expected_std = np.column_stack([expected_std])
    test_measurement = models.Measurement(measurements=test_data, time=test_time)

    test_measurement_std = computations.moving_std(
        test_measurement, epoch_length=EPOCH_LENGTH
    )

    assert np.allclose(test_measurement_std.measurements, expected_std)
    assert test_measurement_std.measurements.shape[1] == test_data.shape[1]
    assert test_measurement_std.time.shape[0] == expected_time_shape


@pytest.mark.parametrize(
    "window_size, expected_output",
    [
        (3, np.array([[5, 3.5, 2], [4, 5, 1], [6.5, 6.5, 1]])),
        (2, np.array([[1, 2, 3], [5, 3.5, 2], [6.5, 6.5, 1]])),
    ],
)
def test_moving_median(window_size: int, expected_output: np.ndarray) -> None:
    """Testing proper function of moving median function."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + timedelta(seconds=i) for i in range(3)]
    dummy_datetime_pl = pl.Series(dummy_datetime_list)
    test_matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [9.0, 5.0, 1.0],
            [4.0, 8.0, 1.0],
        ]
    )

    test_measurement = models.Measurement(
        measurements=test_matrix, time=dummy_datetime_pl
    )

    test_result = computations.moving_median(test_measurement, epoch_length=window_size)

    assert test_result.measurements.shape == expected_output.shape, (
        f"measurements array are not the same shape. Expected {expected_output.shape}, "
        f"instead got: {test_result.measurements.shape}"
    )
    assert np.all(
        np.isclose(test_result.measurements, expected_output)
    ), "Test results do not match the expected output"


def test_resample_downsample_simple() -> None:
    """Test the downsampling happy-path."""
    time = [datetime(1990, 1, 1) + timedelta(seconds=secs) for secs in range(4)]
    measurement = models.Measurement(
        measurements=np.array([1, 2, 3, 4]),
        time=pl.Series(time),
    )
    delta_t = 2
    expected = models.Measurement(
        measurements=np.array([1.5, 3.5]), time=pl.Series("time", time[0::2])
    )

    actual = computations.resample(measurement, delta_t)

    assert np.allclose(actual.measurements, expected.measurements)
    assert all(actual.time == expected.time)


def test_resample_downsample_mismatch_time_start() -> None:
    """Test the downsampling with 60 and 30Hz.

    This test is specifically to catch any issues that arise with sampling
    associated with imprecise detla_t values.
    This test will fail without the `start_by` option in the
    polars`group_by_dynamic` function used by computations.resample.
    """
    time = [
        datetime(1990, 1, 1) + timedelta(microseconds=16666 * idx) for idx in range(100)
    ]
    measurement = models.Measurement(
        measurements=np.ones(len(time)),
        time=pl.Series(time),
    )
    delta_t = 1 / 30
    expected_sampling_rate = 30
    expected_time_length = len(time) / 2
    expected_time = pl.Series(
        [
            time[0] + timedelta(microseconds=delta_t * 1_000_000 * idx)
            for idx in range(round(expected_time_length))
        ]
    )

    actual = computations.resample(measurement, delta_t)
    actual_sampling_rate = 1 / (actual.time[1] - actual.time[0]).total_seconds()

    assert len(actual.time) == expected_time_length
    assert time[0] == actual.time[0]
    assert np.isclose(actual_sampling_rate, expected_sampling_rate)
    assert np.allclose(
        actual.time.dt.timestamp().to_numpy(), expected_time.dt.timestamp().to_numpy()
    )


def test_resample_upsample() -> None:
    """Test the upsampling happy-path."""
    time = [
        datetime(1990, 1, 1) + timedelta(microseconds=16666 * idx) for idx in range(10)
    ]
    measurement = models.Measurement(
        measurements=np.arange(len(time)),
        time=pl.Series(time),
    )
    delta_t = 1 / 120

    expected_length = len(time) * 2 - 1
    expected_time = [
        time[0] + timedelta(microseconds=delta_t * 1_000_000 * idx)
        for idx in range(expected_length)
    ]
    expected = models.Measurement(
        measurements=np.arange(0, expected_length * 0.5, 0.5),
        time=pl.Series("time", expected_time),
    )

    actual = computations.resample(measurement, delta_t)

    assert len(actual.time) == expected_length
    assert all(actual.measurements == expected.measurements)
    assert np.allclose(
        actual.time.dt.timestamp().to_numpy(), expected.time.dt.timestamp().to_numpy()
    )


def test_resample_value_error() -> None:
    """Test that the ValueError is raised for negative delta_t."""
    time = [
        datetime(1990, 1, 1, second=1),
        datetime(1990, 1, 1, second=2),
    ]
    measurement = models.Measurement(
        measurements=np.array([1, 2]),
        time=pl.Series("time", time),
    )
    delta_t = -1

    with pytest.raises(ValueError):
        computations.resample(measurement, delta_t)


def test_nonwear_majority_vote() -> None:
    """Tests the majority vote function for nonwear."""
    time1 = [
        datetime(1990, 1, 1, 1, 1) + timedelta(seconds=60 * secs) for secs in range(900)
    ]
    time3 = [
        datetime(1990, 1, 1, 1, 1, 10) + timedelta(seconds=100 * secs)
        for secs in range(11)
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

    nonwear = computations.majority_vote_non_wear(
        nonwear_cta=nonwear1,
        nonwear_detach=nonwear2,
        nonwear_ggir=nonwear3,
        temporal_resolution=5,
    )

    assert np.all(nonwear.measurements == 1)
