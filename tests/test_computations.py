"""Test the function of computations module."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from wristpy.core import computations, models
from wristpy.io.readers import readers


def test_moving_mean_epoch_length_is_negative() -> None:
    """Test error if the epoch length is negative."""
    tmp_data = np.array([1, 2, 3])
    time = readers.unix_epoch_time_to_polars_datetime(np.array([1, 2, 3]), "s")
    tmp_Measurement = models.Measurement(measurements=tmp_data, time=time)

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
    test_measurement = models.Measurement(measurements=test_data, time=test_time)

    test_measurement_mean = computations.moving_mean(test_measurement, epoch_length=5)

    assert np.allclose(test_measurement_mean.measurements, expected_mean)
    assert test_measurement_mean.measurements.ndim == expected_mean.ndim
    assert np.isclose(test_measurement_mean.time.shape[0], (test_time.shape[0] / 5))


def test_moving_mean_three_columns() -> None:
    """Test the functionality of the moving mean function for three column array."""
    signal_length = 20
    epoch_length = 5
    test_data = np.arange(0, signal_length * 3).reshape(signal_length, 3)
    test_time = readers.unix_epoch_time_to_polars_datetime(
        np.arange(0, signal_length), "s"
    )
    expected_mean = np.array(
        ([[6.0, 7.0, 8.0], [21.0, 22.0, 23.0], [36.0, 37.0, 38.0], [51.0, 52.0, 53.0]])
    )
    test_measurement = models.Measurement(measurements=test_data, time=test_time)

    test_measurement_mean = computations.moving_mean(
        test_measurement, epoch_length=epoch_length
    )

    assert np.allclose(test_measurement_mean.measurements, expected_mean)
    assert test_measurement_mean.measurements.shape[1] == test_data.shape[1]
    assert np.isclose(
        test_measurement_mean.time.shape[0], (test_time.shape[0] / epoch_length)
    )


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
    dummy_datetime_pl = pl.Series("time", dummy_datetime_list)
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

    test_result = computations.moving_median(test_measurement, window_size=window_size)

    assert test_result.measurements.shape == expected_output.shape, (
        f"measurements array are not the same shape. Expected {expected_output.shape}, "
        f"instead got: {test_result.measurements.shape}"
    )
    assert np.all(
        np.isclose(test_result.measurements, expected_output)
    ), "Test results do not match the expected output"
