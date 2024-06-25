"""Test the functionality of the SleepDetection class."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from wristpy.core import models
from wristpy.processing import analytics


@pytest.fixture
def sleep_detection() -> analytics.SleepDetection:
    """Return a SleepDetection instance with dummy data."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + timedelta(seconds=i) for i in range(1000)]
    test_time = pl.Series("time", dummy_datetime_list)
    anglez = np.random.rand(1000, 1)
    non_wear_flag = np.random.randint(2, size=1000)
    anglez_measurement = models.Measurement(measurements=anglez, time=test_time)
    non_wear_measurement = models.Measurement(
        measurements=non_wear_flag, time=test_time
    )
    return analytics.SleepDetection(anglez_measurement, non_wear_measurement, "GGIR")


def test_find_long_blocks(sleep_detection: analytics.SleepDetection) -> None:
    """Test the _find_long_blocks method."""
    below_threshold = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0])
    block_length = 5
    result = sleep_detection._find_long_blocks(below_threshold, block_length)
    expected_result = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.array_equal(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"
