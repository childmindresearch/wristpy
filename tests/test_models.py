"""Testing the watchdata class."""

import numpy as np
import polars as pl
import pytest

from wristpy.core.models import Measurement, WatchData


def unix_epoch_time_converter_to_polars_seconds(time: np.ndarray) -> pl.Series:
    """Convert unix epoch time to polars Series.

    This helper function is specific to the test file using seconds.
    """
    return pl.from_epoch(pl.Series(time), time_unit="s")


def test_watchdata_model_1D_acceleration() -> None:
    """Test the WatchData to catch 1D error in acceleration."""
    sensor_data = np.array([1, 2, 3])
    time = unix_epoch_time_converter_to_polars_seconds(np.array([1, 2, 3]))

    acceleration = Measurement(measurements=sensor_data, time=time)

    with pytest.raises(ValueError):
        WatchData(acceleration=acceleration)


def test_watchdata_model_acceleration_three_columns() -> None:
    """Test the WatchData to catch 3 columns error in acceleration."""
    sensor_data = np.array([[1, 2], [3, 4]])
    time = unix_epoch_time_converter_to_polars_seconds(np.array([1, 2]))

    acceleration = Measurement(measurements=sensor_data, time=time)

    with pytest.raises(ValueError):
        WatchData(acceleration=acceleration)


def test_watchdata_model() -> None:
    """Test the WatchData model."""
    accel_data = np.array([[1, 2, 3], [4, 5, 6]])
    accel_time = unix_epoch_time_converter_to_polars_seconds(np.array([1, 2]))
    sensor_data = np.array([1, 2, 3])
    time = unix_epoch_time_converter_to_polars_seconds(np.array([1, 2, 3]))

    acceleration = Measurement(measurements=accel_data, time=accel_time)
    lux = Measurement(measurements=sensor_data, time=time)
    temp = Measurement(measurements=sensor_data, time=time)

    watch_data = WatchData(acceleration=acceleration, lux=lux, temperature=temp)

    assert np.array_equal(watch_data.acceleration.measurements, accel_data)
    assert np.array_equal(watch_data.lux.measurements, sensor_data)
    assert np.array_equal(watch_data.temperature.measurements, sensor_data)
    assert np.array_equal(
        watch_data.temperature.time.dt.timestamp().to_numpy(),
        np.array([1, 2, 3]) * 1000000,
    )
    assert isinstance(watch_data.battery, type(None))


def test_measurement_model_time_type() -> None:
    """Test the error when time is not a datetime series."""
    time = pl.Series([1, 2, 3])
    with pytest.raises(ValueError):
        Measurement(measurements=np.array([1, 2, 3]), time=time)


def test_measurement_model_time_sorted() -> None:
    """Test the error when time is not sorted."""
    time = unix_epoch_time_converter_to_polars_seconds(np.array([2, 1, 3]))

    with pytest.raises(ValueError):
        Measurement(measurements=np.array([1, 2, 3]), time=time)


def test_measurement_model() -> None:
    """Test the Measurement model."""
    time = unix_epoch_time_converter_to_polars_seconds(np.array([1, 2, 3]))

    measurement = Measurement(measurements=np.array([1, 2, 3]), time=time)

    assert np.array_equal(measurement.measurements, np.array([1, 2, 3]))
    assert np.array_equal(
        measurement.time.dt.timestamp().to_numpy(),
        np.array([1, 2, 3]) * 1000000,
    )
