"""Testing the watchdata class."""

import numpy as np
import polars as pl
import pytest

from wristpy.core.models import Measurement, WatchData


@pytest.fixture
def create_mock_measurement() -> Measurement:
    """Create a mock measurement instance."""

    def _create_mock_measurement(
        measurements: np.ndarray, time: np.ndarray
    ) -> Measurement:
        time = time * 1000000
        time_series = pl.from_epoch(pl.Series(time), time_unit="us")
        return Measurement(measurements=measurements, time=time_series)

    return _create_mock_measurement


def test_watchdata_model_1D_acceleration(create_mock_measurement: Measurement) -> None:
    """Test the WatchData to catch 1D error in acceleration."""
    sensor_data = np.array([1, 2, 3])
    time = np.array([1704110400, 1704110401, 1704110402])
    with pytest.raises(ValueError):
        acceleration = create_mock_measurement(sensor_data, time)
        WatchData(acceleration=acceleration)


def test_watchdata_model_acceleration_three_columns(
    create_mock_measurement: Measurement,
) -> None:
    """Test the WatchData to catch 3 columns error in acceleration."""
    with pytest.raises(ValueError):
        acceleration = create_mock_measurement(
            np.array([[1, 2], [4, 5]]), np.array([1, 2])
        )
        WatchData(acceleration=acceleration)


def test_watchdata_model(
    create_mock_measurement: Measurement,
) -> None:
    """Test the WatchData model."""
    accel_data = np.array([[1, 2, 3], [4, 5, 6]])
    accel_time = np.array([1704110400, 1704110401])
    sensor_data = np.array([1, 2, 3])
    time = np.array([1704110400, 1704110401, 1704110402])
    acceleration = create_mock_measurement(accel_data, accel_time)
    lux = create_mock_measurement(sensor_data, time)
    temp = create_mock_measurement(sensor_data, time)

    # Create a WatchData instance
    watch_data = WatchData(acceleration=acceleration, lux=lux, temperature=temp)

    # Assert that the data was stored correctly
    assert np.array_equal(watch_data.acceleration.measurements, accel_data)
    assert np.array_equal(watch_data.lux.measurements, sensor_data)
    assert np.array_equal(watch_data.temperature.measurements, sensor_data)
    assert np.array_equal(
        watch_data.temperature.time.dt.timestamp().to_numpy(),
        time * 1000000,
    )
