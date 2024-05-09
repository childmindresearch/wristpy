"""Test the function of computations module."""

import numpy as np
import pytest

from wristpy.core import computations
from wristpy.core.models import Measurement
from wristpy.io.readers import readers


def test_moving_mean_epoch_length_is_int() -> None:
    """Test error for non-integer epoch length."""
    tmp_data = np.array([1, 2, 3])
    time = readers.unix_epoch_time_to_polars_datetime(np.array([1, 2, 3]), "s")
    tmp_Measurement = Measurement(measurements=tmp_data, time=time)

    with pytest.raises(ValueError):
        computations.moving_mean(tmp_Measurement, epoch_length=5.5)


def test_moving_mean_epoch_length_is_negative() -> None:
    """Test error if the epoch length is negative."""
    tmp_data = np.array([1, 2, 3])
    time = readers.unix_epoch_time_to_polars_datetime(np.array([1, 2, 3]), "s")
    tmp_Measurement = Measurement(measurements=tmp_data, time=time)

    with pytest.raises(ValueError):
        computations.moving_mean(tmp_Measurement, epoch_length=-5)


def test_moving_mean_one_column() -> None:
    """Test the functionality of the moving mean function for 1D Measurement."""
    signal_length = 20
    test_data = np.arange(0, signal_length)
    test_time = readers.unix_epoch_time_to_polars_datetime(
        np.arange(0, signal_length), "s"
    )
    expected_mean = np.array([2.0, 7.0, 12.0, 17.0])
    test_measurement = Measurement(measurements=test_data, time=test_time)

    test_measurement_mean = computations.moving_mean(test_measurement, epoch_length=5)

    assert np.allclose(test_measurement_mean.measurements, expected_mean)


def test_moving_mean_three_columns() -> None:
    """Test the functionality of the moving mean function for 3D Measurement."""
    signal_length = 20
    test_data = np.arange(0, signal_length * 3).reshape(signal_length, 3)
    test_time = readers.unix_epoch_time_to_polars_datetime(
        np.arange(0, signal_length), "s"
    )
    expected_mean = np.array(
        ([[6.0, 7.0, 8.0], [21.0, 22.0, 23.0], [36.0, 37.0, 38.0], [51.0, 52.0, 53.0]])
    )
    test_measurement = Measurement(measurements=test_data, time=test_time)

    test_measurement_mean = computations.moving_mean(test_measurement, epoch_length=5)

    assert np.allclose(test_measurement_mean.measurements, expected_mean)
    assert test_measurement_mean.measurements.shape[1] == test_data.shape[1]
