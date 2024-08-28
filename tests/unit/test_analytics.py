"""Test the functionality of the GGIRSleepDetection class."""

import datetime
import math
from typing import List

import numpy as np
import polars as pl
import pytest

from wristpy.core import models
from wristpy.processing import analytics


@pytest.fixture
def sleep_detection() -> analytics.GGIRSleepDetection:
    """Return a GGIRSleepDetection instance with dummy data."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_datetime_list = [
        dummy_date + datetime.timedelta(seconds=i) for i in range(3600)
    ]
    test_time = pl.Series("time", dummy_datetime_list)
    anglez = np.random.randint(-90, 90, size=3600)
    anglez_measurement = models.Measurement(measurements=anglez, time=test_time)
    return analytics.GGIRSleepDetection(anglez_measurement)


def test_fill_false_blocks(
    sleep_detection: analytics.GGIRSleepDetection,
) -> None:
    """Test the _fill_false_short_blocks method."""
    gap_block = 3
    sleep_idx_array = np.array(
        [True, False, True, False, False, False, True, True, False, True, False, True]
    )
    expected_result = np.array(
        [True, True, True, False, False, False, True, True, True, True, True, True]
    )

    result = sleep_detection._fill_false_blocks(sleep_idx_array, gap_block)

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


def test_find_periods() -> None:
    """Test the _find_periods method."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_datetime_list = [
        dummy_date + datetime.timedelta(seconds=i) for i in range(10)
    ]
    test_time = pl.Series("time", dummy_datetime_list)
    window_data = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 1])
    window_measurement = models.Measurement(measurements=window_data, time=test_time)
    expected_result = [
        (dummy_datetime_list[1], dummy_datetime_list[3]),
        (dummy_datetime_list[5], dummy_datetime_list[5]),
        (dummy_datetime_list[9], dummy_datetime_list[9]),
    ]

    result = analytics._find_periods(window_measurement)

    assert result == expected_result, f"Expected {expected_result}, but got {result}"


@pytest.mark.parametrize(
    "non_wear_array, expected_result",
    [
        (np.array([0, 1, 1, 1, 0]), []),
        (np.array([0, 0, 0, 1, 1]), []),
        (np.array([0, 0, 0, 1, 0]), []),
    ],
)
def test_remove_nonwear_periods_overlap(
    non_wear_array: np.ndarray, expected_result: List
) -> None:
    """Test the _remove_nonwear_from_sleep method.

    This test is for the following cases:
        - where the non-wear period overlaps with sleep onset.
        - where the non-wear period overlaps with sleep wakeup.
        - where the non-wear period is within the sleep window.
    """
    dummy_date = datetime.datetime(2024, 5, 2)
    sleep_windows = [
        analytics.SleepWindow(
            onset=dummy_date + datetime.timedelta(hours=2),
            wakeup=dummy_date + datetime.timedelta(hours=4),
        )
    ]
    non_wear_time = pl.Series(
        "time",
        [dummy_date + datetime.timedelta(hours=i) for i in range(len(non_wear_array))],
    )
    non_wear_measurement = models.Measurement(
        measurements=non_wear_array, time=non_wear_time
    )

    result = analytics.remove_nonwear_from_sleep(non_wear_measurement, sleep_windows)

    assert result == expected_result, f"Expected {expected_result}, but got {result}"


def test_remove_nonwear_periods_no_overlap() -> None:
    """Test the _remove_nonwear_from_sleep method.

    This test is for the case where the non-wear period does not overlap
    with the sleep window.
    """
    dummy_date = datetime.datetime(2024, 5, 2)
    sleep_windows = [
        analytics.SleepWindow(
            onset=dummy_date + datetime.timedelta(hours=2),
            wakeup=dummy_date + datetime.timedelta(hours=4),
        )
    ]

    non_wear_array = np.array([1, 0, 0, 0, 0])
    non_wear_time = pl.Series(
        "time",
        [dummy_date + datetime.timedelta(hours=i) for i in range(len(non_wear_array))],
    )
    non_wear_measurement = models.Measurement(
        measurements=non_wear_array, time=non_wear_time
    )
    expected_result = sleep_windows

    result = analytics.remove_nonwear_from_sleep(non_wear_measurement, sleep_windows)

    assert result == expected_result, f"Expected {expected_result}, but got {result}"


def test_spt_window(sleep_detection: analytics.GGIRSleepDetection) -> None:
    """Test the _spt_window method."""
    sleep_detection.anglez.measurements = np.zeros(
        len(sleep_detection.anglez.measurements)
    )
    expected_length = int(len(sleep_detection.anglez.measurements) / 5) - 1
    expected_result = np.ones(expected_length)

    result = sleep_detection._spt_window(sleep_detection.anglez)

    assert np.array_equal(
        result.measurements, expected_result
    ), f"Expected {expected_result}, but got {result.measurements}"
    assert np.array_equal(
        len(result.time), expected_length
    ), f"Expected {expected_length}, but got {len(result.time)}"


def test_spt_window_null(sleep_detection: analytics.GGIRSleepDetection) -> None:
    """Test the _spt_window method."""
    expected_length = int(len(sleep_detection.anglez.measurements) / 5) - 1
    expected_result = np.zeros(expected_length)

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
    dummy_date = datetime.datetime(2024, 5, 2)
    spt_periods = [
        (
            dummy_date + datetime.timedelta(hours=1),
            dummy_date + datetime.timedelta(hours=3),
        )
    ]
    sib_periods = [
        (
            dummy_date + datetime.timedelta(hours=2),
            dummy_date + datetime.timedelta(hours=4),
        )
    ]
    expected_output = analytics.SleepWindow(
        onset=dummy_date + datetime.timedelta(hours=2),
        wakeup=dummy_date + datetime.timedelta(hours=4),
    )

    result = sleep_detection._find_onset_wakeup_times(spt_periods, sib_periods)

    assert result[0].onset == expected_output.onset
    assert result[0].wakeup == expected_output.wakeup


def test_run_sleep_detection(sleep_detection: analytics.GGIRSleepDetection) -> None:
    """Test the full sleep detection process."""
    result = sleep_detection.run_sleep_detection()

    assert result == []
    assert isinstance(result, List)


def test_physical_activity_thresholds_unsorted() -> None:
    """Test the physical_activity_thresholds method with unsorted thresholds."""
    tmp_data = np.array([1, 2, 3])
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + datetime.timedelta(seconds=i) for i in range(3)]
    time = pl.Series("time", dummy_datetime_list)
    tmp_measurement = models.Measurement(measurements=tmp_data, time=time)
    thresholds = (3, 2, 1)

    with pytest.raises(ValueError):
        analytics.compute_physical_activty_categories(tmp_measurement, thresholds)


def test_physical_activity_thresholds() -> None:
    """Test the physical_activity_thresholds method."""
    tmp_data = np.arange(4)
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + datetime.timedelta(seconds=i) for i in range(4)]
    time = pl.Series("time", dummy_datetime_list)
    tmp_measurement = models.Measurement(measurements=tmp_data, time=time)
    thresholds = (0, 1, 2)
    expected_result = np.array(
        [
            0,
            1,
            2,
            3,
        ]
    )

    result = analytics.compute_physical_activty_categories(tmp_measurement, thresholds)

    assert np.array_equal(result.measurements, expected_result)
    assert np.array_equal(result.time, time)
