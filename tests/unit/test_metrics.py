"""Testing functions of metrics module."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from wristpy.core import models
from wristpy.processing import metrics


def test_rolling_median_wrong_ndims() -> None:
    """Test raising value error when ndims is 1 or fewer."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime = pl.Series("time", [dummy_date])
    bad_array = np.array([1, 2, 3])
    test_measurement = models.Measurement(measurements=bad_array, time=dummy_datetime)

    with pytest.raises(ValueError):
        metrics.rolling_median(test_measurement)


def test_rolling_median_window_size() -> None:
    """Test the window size being <=1."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime = pl.Series("time", [dummy_date])
    test_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    test_measurement = models.Measurement(measurements=test_array, time=dummy_datetime)

    with pytest.raises(ValueError):
        metrics.rolling_median(test_measurement, window_size=0)


@pytest.mark.parametrize(
    "window_size, expected_output",
    [
        (3, np.array([[5, 3.5, 2], [4, 5, 1], [6.5, 6.5, 1]])),
        (2, np.array([[5, 3.5, 2], [4, 5, 1], [6.5, 6.5, 1]])),
    ],
)
def test_rolling_median(window_size: int, expected_output: np.ndarray) -> np.ndarray:
    """Testing proper function of rolling median function."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime = pl.Series("time", [dummy_date] * 3)
    test_matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [9.0, 5.0, 1.0],
            [4.0, 8.0, 1.0],
        ]
    )

    test_measurement = models.Measurement(measurements=test_matrix, time=dummy_datetime)

    test_result = metrics.rolling_median(test_measurement, window_size=window_size)

    assert test_result.measurements.shape == expected_output.shape, (
        f"measurements array are not the same shape. Expected {expected_output.shape}, "
        f"instead got: {test_result.measurements.shape}"
    )
    assert np.all(
        np.isclose(test_result.measurements, expected_output)
    ), "test_result values and expected_output values are not the same."
