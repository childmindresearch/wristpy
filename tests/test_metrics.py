"""Test the function of metrics module."""

import numpy as np
import pytest

from wristpy.core.models import Measurement
from wristpy.io.readers import readers
from wristpy.processing import metrics


def test_moving_mean_is_measurement() -> None:
    """Test that the moving mean function returns a Measurement object."""
    test_file = "tests/sample_data/example_actigraph.gt3x"
    watch_data = readers.read_watch_data(test_file)

    moving_mean = metrics.moving_mean(watch_data.acceleration)

    assert isinstance(moving_mean, Measurement)


def test_moving_mean_is_not_measurement() -> None:
    """Test error for non-measurement input."""
    with pytest.raises(ValueError):
        metrics.moving_mean(np.array([1, 2, 3, 4, 5]))


def test_moving_mean_epoch_length_is_int() -> None:
    """Test error for non-integer epoch length."""
    test_file = "tests/sample_data/example_actigraph.gt3x"

    watch_data = readers.read_watch_data(test_file)

    with pytest.raises(ValueError):
        metrics.moving_mean(watch_data.acceleration, epoch_length=5.5)


def test_moving_mean_epoch_length_is_negative() -> None:
    """Test error if the epoch length is negative."""
    test_file = "tests/sample_data/example_actigraph.gt3x"

    watch_data = readers.read_watch_data(test_file)

    with pytest.raises(ValueError):
        metrics.moving_mean(watch_data.acceleration, epoch_length=-5)


def test_moving_mean() -> None:
    """Test the functionality of the moving mean function."""
    signal_length = 20
    test_data = np.arange(0, signal_length)
    test_time = readers.unix_epoch_time_to_polars_datetime(
        np.arange(0, signal_length), "s"
    )

    test_measurement = Measurement(measurements=test_data, time=test_time)
    test_measurement_mean = metrics.moving_mean(test_measurement, epoch_length=5)

    assert np.array_equal(
        test_measurement_mean.measurements.flatten(), np.array([2.0, 7.0, 12.0, 17.0])
    )
