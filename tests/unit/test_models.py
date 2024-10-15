"""Testing the watchdata class."""

import numpy as np
import polars as pl
import pytest
from polars import testing

from wristpy.core import models
from wristpy.io.readers import readers


def test_watchdata_model_1D_acceleration() -> None:
    """Test the WatchData to catch 1D error in acceleration."""
    sensor_data = np.array([1, 2, 3])
    time = readers.unix_epoch_time_to_polars_datetime(np.array([1, 2, 3]), "s")

    acceleration = models.Measurement(measurements=sensor_data, time=time)

    with pytest.raises(ValueError):
        models.WatchData(acceleration=acceleration)


def test_watchdata_model_acceleration_three_columns() -> None:
    """Test the WatchData to catch 3 columns error in acceleration."""
    sensor_data = np.array([[1, 2], [3, 4]])
    time = readers.unix_epoch_time_to_polars_datetime(np.array([1, 2]), "s")

    acceleration = models.Measurement(measurements=sensor_data, time=time)

    with pytest.raises(ValueError):
        models.WatchData(acceleration=acceleration)


def test_watchdata_model() -> None:
    """Test the WatchData model."""
    accel_data = np.array([[1, 2, 3], [4, 5, 6]])
    accel_time = readers.unix_epoch_time_to_polars_datetime(np.array([1, 2]), "s")
    sensor_data = np.array([1, 2, 3])
    time = readers.unix_epoch_time_to_polars_datetime(np.array([1, 2, 3]), "s")

    acceleration = models.Measurement(measurements=accel_data, time=accel_time)
    lux = models.Measurement(measurements=sensor_data, time=time)
    temp = models.Measurement(measurements=sensor_data, time=time)

    watch_data = models.WatchData(acceleration=acceleration, lux=lux, temperature=temp)

    if watch_data.lux is not None:
        assert np.array_equal(watch_data.lux.measurements, sensor_data)
    if watch_data.temperature is not None:
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
        models.Measurement(measurements=np.array([1, 2, 3]), time=time)


def test_measurement_model_time_unique() -> None:
    """Test the error when time is not unique."""
    time = readers.unix_epoch_time_to_polars_datetime(np.array([1, 1, 3]), "s")

    with pytest.raises(ValueError):
        models.Measurement(measurements=np.array([1, 2, 3]), time=time)


def test_measurement_model_time_sorted() -> None:
    """Test the error when time is not sorted."""
    time = readers.unix_epoch_time_to_polars_datetime(np.array([2, 1, 3]), "s")

    with pytest.raises(ValueError):
        models.Measurement(measurements=np.array([1, 2, 3]), time=time)


def test_measurement_model_time_empty() -> None:
    """Test the error when time is empty."""
    time = readers.unix_epoch_time_to_polars_datetime(np.array([]), "s")

    with pytest.raises(ValueError):
        models.Measurement(measurements=np.array([1, 2, 3]), time=time)


def test_measurement_model_measurements_empty() -> None:
    """Test the error when measurements is empty."""
    time = readers.unix_epoch_time_to_polars_datetime(np.array([1, 2, 3]), "s")

    with pytest.raises(ValueError):
        models.Measurement(measurements=np.array([]), time=time)


def test_measurement_model() -> None:
    """Test the Measurement model.

    Note that the polars.dt.timestamp() method does not support returns in seconds,
    the default is microseconds, thus we multiple by 1e6 in the comparison.
    """
    time = readers.unix_epoch_time_to_polars_datetime(np.array([1, 2, 3]), "s")

    measurement = models.Measurement(measurements=np.array([1, 2, 3]), time=time)

    assert np.array_equal(measurement.measurements, np.array([1, 2, 3]))

    assert np.array_equal(
        measurement.time.dt.timestamp().to_numpy(),
        np.array([1, 2, 3]) * 1000000,
    )


def test_measurement_from_data_frame() -> None:
    """Tests conversion of a DataFrame to a Measurement."""
    time = readers.unix_epoch_time_to_polars_datetime(np.array([1]), "s")
    df = pl.DataFrame({"time": time, "x": np.array([1])})

    actual = models.Measurement.from_data_frame(df)

    assert np.all(actual.measurements == df["x"].to_numpy())
    testing.assert_series_equal(actual.time, df["time"])


def test_measurement_lazy_frame() -> None:
    """Tests conversion of a Measurement to a LazyFrame."""
    time = readers.unix_epoch_time_to_polars_datetime(np.array([1, 2, 3]), "s")
    measurement = models.Measurement(measurements=np.array([1, 2, 3]), time=time)

    actual = measurement.lazy_frame().collect()

    assert np.all(actual.drop("time").to_numpy().squeeze() == measurement.measurements)
    testing.assert_series_equal(actual["time"], measurement.time)
