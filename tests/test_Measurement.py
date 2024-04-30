"""Testing the measurement model class."""

import numpy as np
import polars as pl
import pytest

from wristpy.core.models import Measurement


@pytest.mark.parametrize(
    "sensor_data, time_data, should_raise",
    [
        (np.array([1, 2, 3]), np.array([1704110400, 1704110401, 1704110402]), False),
        (np.array([1, 2, 3]), np.array([1704110400, 1704110401, 1704110402]), True),
    ],
)
def test_measurement_model(
    sensor_data: np.ndarray, time_data: np.ndarray | pl.Series, should_raise: bool
) -> None:
    """Test the ability to create a Measurement instance.

    Args:
        sensor_data: The sensor data.
        time_data: The time data.
        should_raise: Whether an error should be raised.
    """
    if should_raise:
        with pytest.raises(ValueError):
            Measurement(measurements=sensor_data, time=time_data)
        return
    # Assert that the data was stored correctly
    else:
        # Convert time data to a polars series of datetime in us
        time_data = time_data * 1000000
        time_data_datetime = pl.from_epoch(pl.Series(time_data), time_unit="us")

        measurement = Measurement(measurements=sensor_data, time=time_data_datetime)

        assert np.array_equal(measurement.measurements, sensor_data)
        assert np.array_equal(measurement.time.dt.timestamp().to_numpy(), time_data)
