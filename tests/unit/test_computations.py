"""Test the function of computations module."""


import numpy as np
import pytest

from wristpy.core import computations, models
from wristpy.io.readers import readers

SIGNAL_LENGTH = 20
EPOCH_LENGTH = 5


def test_moving_mean_epoch_length_is_negative() -> None:
    """Test error if the epoch length is negative."""
    tmp_data = np.array([1, 2, 3])
    time = readers.unix_epoch_time_to_polars_datetime(np.array([1, 2, 3]), "s")
    tmp_Measurement = models.Measurement(measurements=tmp_data, time=time)

    with pytest.raises(ValueError):
        computations.moving_mean(tmp_Measurement, epoch_length=-EPOCH_LENGTH)


def test_moving_mean_one_column() -> None:
    """Test the functionality of the moving mean function for 1D Measurement."""
    test_data = np.arange(0, SIGNAL_LENGTH)
    test_time = readers.unix_epoch_time_to_polars_datetime(
        np.arange(0, SIGNAL_LENGTH), "s"
    )
    expected_time_shape = test_time.shape[0] / EPOCH_LENGTH
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
    test_time = readers.unix_epoch_time_to_polars_datetime(
        np.arange(0, SIGNAL_LENGTH), "s"
    )
    expected_time_shape = test_time.shape[0] / EPOCH_LENGTH
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
    time = readers.unix_epoch_time_to_polars_datetime(np.array([1, 2, 3]), "s")
    tmp_Measurement = models.Measurement(measurements=tmp_data, time=time)

    with pytest.raises(ValueError):
        computations.moving_std(tmp_Measurement, epoch_length=-EPOCH_LENGTH)


@pytest.mark.parametrize(
    "test_array, expected_std",
    [
        (np.ones(SIGNAL_LENGTH), np.array([0.0, 0.0, 0.0, 0.0])),
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
    test_time = readers.unix_epoch_time_to_polars_datetime(
        np.arange(0, SIGNAL_LENGTH), "s"
    )
    expected_time_shape = int(test_time.shape[0] / EPOCH_LENGTH)
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
        (np.ones(SIGNAL_LENGTH), np.array([0.0, 0.0, 0.0, 0.0])),
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
    test_data = np.column_stack([test_data] * 3)
    test_time = readers.unix_epoch_time_to_polars_datetime(
        np.arange(0, SIGNAL_LENGTH), "s"
    )
    expected_time_shape = int(test_time.shape[0] / EPOCH_LENGTH)
    expected_std = np.column_stack([expected_std])
    test_measurement = models.Measurement(measurements=test_data, time=test_time)

    test_measurement_std = computations.moving_std(
        test_measurement, epoch_length=EPOCH_LENGTH
    )

    assert np.allclose(test_measurement_std.measurements, expected_std)
    assert test_measurement_std.measurements.shape[1] == test_data.shape[1]
    assert test_measurement_std.time.shape[0] == expected_time_shape
