"""Test the functionality of the GGIRSleepDetection class."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from wristpy.core import models
from wristpy.processing import analytics


@pytest.fixture
def sleep_detection() -> analytics.GGIRSleepDetection:
    """Return a GGIRSleepDetection instance with dummy data."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + timedelta(seconds=i) for i in range(3600)]
    test_time = pl.Series("time", dummy_datetime_list)
    anglez = np.random.uniform(-90, 90, size=3600)
    anglez_measurement = models.Measurement(measurements=anglez, time=test_time)
    return analytics.GGIRSleepDetection(anglez_measurement)


def test_find_long_blocks(
    sleep_detection: analytics.GGIRSleepDetection,
) -> None:
    """Test the _find_long_blocks method."""
    block_length = 3
    below_threshold = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    expected_result = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    result = sleep_detection._find_long_blocks(below_threshold, block_length)

    assert np.array_equal(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"


def test_fill_short_blocks(
    sleep_detection: analytics.GGIRSleepDetection,
) -> None:
    """Test the _fill_short_blocks method."""
    gap_block = 3
    sleep_idx_array = np.array([0, 0, 0, 1, 1, 0, 0, 1])
    expected_result = np.array([0, 0, 0, 1, 1, 1, 1, 1])

    result = sleep_detection._fill_short_blocks(sleep_idx_array, gap_block)

    assert np.array_equal(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"


def test_compute_abs_diff_mean_anglez(
    sleep_detection: analytics.GGIRSleepDetection,
) -> None:
    """Test the _compute_abs_diff_mean_anglez method."""
    sleep_detection.anglez.measurements = np.ones(
        len(sleep_detection.anglez.measurements)
    )
    expected_length = (
        math.ceil(len(sleep_detection.anglez.measurements) / 5) - 1
    )  # 5 is the default epoch length, -1 because diff happens after the moving_mean
    expected_result = np.zeros(expected_length)

    result = sleep_detection._compute_abs_diff_mean_anglez(sleep_detection.anglez)

    assert np.array_equal(
        result.measurements, expected_result
    ), f"Expected {expected_result}, but got {result.measurements}"
    assert np.array_equal(len(result.time), expected_length)


def test_find_periods(
    sleep_detection: analytics.GGIRSleepDetection,
) -> None:
    """Test the _find_periods method."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + timedelta(seconds=i) for i in range(10)]
    test_time = pl.Series("time", dummy_datetime_list)
    window_data = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 1])
    window_measurement = models.Measurement(measurements=window_data, time=test_time)
    expected_result = [
        (dummy_datetime_list[1], dummy_datetime_list[3]),
        (dummy_datetime_list[5], dummy_datetime_list[5]),
        (dummy_datetime_list[9], dummy_datetime_list[9]),
    ]

    result = sleep_detection._find_periods(window_measurement)

    assert result == expected_result, f"Expected {expected_result}, but got {result}"


@pytest.mark.parametrize("modifier", [0, 1])
def test_spt_window(
    sleep_detection: analytics.GGIRSleepDetection, modifier: int
) -> None:
    """Test the _spt_window method."""
    half_long_block = 180
    expected_length = int(len(sleep_detection.anglez.measurements) / 5) - 1
    expected_result = np.zeros(expected_length)

    if modifier:
        result = sleep_detection._spt_window(sleep_detection.anglez)
    else:
        sleep_detection.anglez.measurements = np.zeros(
            len(sleep_detection.anglez.measurements)
        )
        expected_result[half_long_block:] = 1
        result = sleep_detection._spt_window(sleep_detection.anglez)

    assert np.array_equal(
        result.measurements, expected_result
    ), f"Expected {expected_result}, but got {result.measurements}"
    assert np.array_equal(
        len(result.time), expected_length
    ), f"Expected {expected_length}, but got {len(result.time)}"


def test_calculate_sib_periods(sleep_detection: analytics.GGIRSleepDetection) -> None:
    """Test the _calculate_sib_periods method."""
    expected_length = math.ceil(len(sleep_detection.anglez.measurements) / 300)
    expected_result = np.zeros(expected_length)

    result = sleep_detection._calculate_sib_periods(sleep_detection.anglez, 10)

    assert np.array_equal(
        result.measurements, expected_result
    ), f"Expected {expected_result}, but got {result.measurements}"
    assert (
        len(result.measurements) == expected_length
    ), f"Expected {expected_length}, but got {len(result.measurements)}"


def test_find_onset_wakeup_times(sleep_detection: analytics.GGIRSleepDetection) -> None:
    """Test the _find_onset_wakeup_times method."""
    dummy_date = datetime(2024, 5, 2)
    spt_periods = [
        (
            dummy_date + timedelta(hours=1),
            dummy_date + timedelta(hours=3),
        )
    ]
    sib_periods = [
        (
            dummy_date + timedelta(hours=2),
            dummy_date + timedelta(hours=4),
        )
    ]
    expected_output = analytics.SleepWindow(
        onset=dummy_date + timedelta(hours=2),
        wakeup=dummy_date + timedelta(hours=4),
    )

    result = sleep_detection._find_onset_wakeup_times(spt_periods, sib_periods)

    assert result[0].onset == expected_output.onset
    assert result[0].wakeup == expected_output.wakeup


def test_run_sleep_detection(sleep_detection: analytics.GGIRSleepDetection) -> None:
    """Test the full sleep detection process."""
    result = sleep_detection.run_sleep_detection()

    assert isinstance(
        result[0], analytics.SleepWindow
    ), "result is not an instance of SleepWindow"
    assert result[0].onset == []
    assert result[0].wakeup == []
