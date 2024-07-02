"""Test the functionality of the SleepDetection class."""

import math
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
    anglez = np.random.uniform(-90, 90, size=1000)
    non_wear_flag = np.random.randint(2, size=1000)
    anglez_measurement = models.Measurement(measurements=anglez, time=test_time)
    non_wear_measurement = models.Measurement(
        measurements=non_wear_flag, time=test_time
    )
    return analytics.SleepDetection(anglez_measurement, non_wear_measurement, "GGIR")


@pytest.mark.parametrize(
    "below_threshold, expected_result",
    [
        (
            np.array([0, 0, 1, 1, 1, 1, 0, 0]),
            np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        ),
        (
            np.array([0, 0, 0, 1, 1, 0, 0, 1]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_find_long_blocks(
    sleep_detection: analytics.SleepDetection,
    below_threshold: np.ndarray,
    expected_result: np.ndarray,
) -> None:
    """Test the _find_long_blocks method."""
    block_length = 3
    result = sleep_detection._find_long_blocks(below_threshold, block_length)
    assert np.array_equal(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"


@pytest.mark.parametrize(
    "sleep_idx_array, expected_result",
    [
        (
            np.array([0, 0, 1, 0, 0, 1, 0, 0]),
            np.array([0, 0, 1, 1, 1, 1, 1, 1]),
        ),
        (
            np.array([0, 0, 0, 1, 1, 0, 0, 1]),
            np.array([0, 0, 0, 1, 1, 1, 1, 1]),
        ),
    ],
)
def test_fill_short_blocks(
    sleep_detection: analytics.SleepDetection,
    sleep_idx_array: np.ndarray,
    expected_result: np.ndarray,
) -> None:
    """Test the _fill_short_blocks method."""
    gap_block = 3
    result = sleep_detection._fill_short_blocks(sleep_idx_array, gap_block)
    assert np.array_equal(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"


def test_compute_abs_diff_mean_anglez(
    sleep_detection: analytics.SleepDetection,
) -> None:
    """Test the _compute_abs_diff_mean_anglez method."""
    sleep_detection.anglez.measurements = np.ones(
        len(sleep_detection.anglez.measurements)
    )
    result = sleep_detection._compute_abs_diff_mean_anglez(sleep_detection.anglez)
    expected_length = (
        math.ceil(len(sleep_detection.anglez.measurements) / 5) - 1
    )  # 5 is the default epoch length, -1 because diff happens after the moving_mean
    expected_result = np.zeros(expected_length)

    assert np.array_equal(
        result.measurements, expected_result
    ), f"Expected {expected_result}, but got {result.measurements}"
    assert np.array_equal(len(result.time), expected_length)
