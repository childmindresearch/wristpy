"""Testing the measurement model class."""

import numpy as np
import polars as pl
import pytest

from wristpy.core.models import Measurement


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


def test_measurement_model_time_type() -> None:
    """Test the error when time is not a datetime series."""
    with pytest.raises(ValueError):
        time = pl.Series([1, 2, 3])
        Measurement(measurements=np.array([1, 2, 3]), time=time)


def test_measurement_model_time_sorted() -> None:
    """Test the error when time is not sorted."""
    with pytest.raises(ValueError):
        time = np.array([1704110409, 1704110410, 1704110403])
        time = time * 1000000
        time_series = pl.from_epoch(pl.Series(time), time_unit="us")
        Measurement(measurements=np.array([1, 2, 3]), time=time_series)


def test_measurement_model(create_mock_measurement: Measurement) -> None:
    """Test the Measurement model."""
    measurement = create_mock_measurement(
        np.array([1, 2, 3]), np.array([1704110400, 1704110401, 1704110402])
    )
    assert np.array_equal(measurement.measurements, np.array([1, 2, 3]))
    assert np.array_equal(
        measurement.time.dt.timestamp().to_numpy(),
        np.array([1704110400, 1704110401, 1704110402]) * 1000000,
    )
