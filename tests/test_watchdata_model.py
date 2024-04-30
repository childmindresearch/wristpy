"""Testing the watchdata class."""

import numpy as np
import polars as pl
import pytest

from wristpy.core.models import Measurement, WatchData


def create_mock_measurement(measurements: np.ndarray, time: np.ndarray) -> Measurement:
    """Create a mock measurement instance.

    Args:
        measurements: Array of measurements.
        time: Array of time values.
    """
    time = time * 1000000
    time_series = pl.from_epoch(pl.Series(time), time_unit="us")
    return Measurement(measurements=measurements, time=time_series)


@pytest.mark.parametrize(
    "acceleration_measurements, acceleration_time, lux_measurements, lux_time, temperature_measurements, temperature_time",  # noqa: E501
    [
        (
            np.array([1, 2, 3]),
            np.array([1704110400, 1704110401, 1704110402]),
            np.array([1, 1, 3000]),
            np.array([1704110409, 1704110410, 1704110403]),
            np.array([27.2, 24.1, 30]),
            np.array([1704110404, 1704110405, 1704110406]),
        ),
        (
            np.array([0.1, -1, 1]),
            np.array([1704110400, 1704110401, 1704110402]),
            np.array([-1, 10.3, 3000]),
            np.array([1704110403, 1704110404, 1704110405]),
            np.array([27.2, 24.5, 36]),
            np.array([1704110406, 1704110407, 1704110408]),
        ),
    ],
)
def test_watchdata_model(
    acceleration_measurements: np.ndarray,
    acceleration_time: np.ndarray,
    lux_measurements: np.ndarray,
    lux_time: np.ndarray,
    temperature_measurements: np.ndarray,
    temperature_time: np.ndarray,
) -> None:
    """Test the WatchData model.

    Args:
        acceleration_measurements: Array of acceleration measurements.
        acceleration_time: pl.Series of datetime .
        lux_measurements: Array of lux measurements.
        lux_time: Array of lux time values.
        temperature_measurements: Array of temperature measurements.
        temperature_time: Array of temperature time values.
    """
    acceleration = create_mock_measurement(acceleration_measurements, acceleration_time)
    lux = create_mock_measurement(lux_measurements, lux_time)
    temp = create_mock_measurement(temperature_measurements, temperature_time)

    # Create a WatchData instance
    watch_data = WatchData(acceleration=acceleration, lux=lux, temperature=temp)

    # Assert that the data was stored correctly
    assert np.array_equal(
        watch_data.acceleration.measurements, acceleration_measurements
    )
    assert np.array_equal(watch_data.lux.measurements, lux_measurements)
    assert np.array_equal(watch_data.temperature.measurements, temperature_measurements)
    assert np.array_equal(
        watch_data.temperature.time.dt.timestamp().to_numpy(),
        temperature_time * 1000000,
    )
