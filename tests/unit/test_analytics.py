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
def sleep_detection() -> analytics.GgirSleepDetection:
    """Return a GGIRSleepDetection instance with dummy data."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_datetime_list = [
        dummy_date + datetime.timedelta(seconds=i) for i in range(3600)
    ]
    test_time = pl.Series("time", dummy_datetime_list)
    anglez = np.random.randint(-90, 90, size=3600)
    anglez_measurement = models.Measurement(measurements=anglez, time=test_time)
    return analytics.GgirSleepDetection(anglez_measurement)


def test_fill_false_blocks() -> None:
    """Test the _fill_false_short_blocks method."""
    gap_block = 3
    sleep_idx_array = np.array(
        [True, False, True, False, False, False, True, True, False, True, False, True]
    )
    expected_result = np.array(
        [True, True, True, False, False, False, True, True, True, True, True, True]
    )

    result = analytics._fill_false_blocks(sleep_idx_array, gap_block)

    assert np.array_equal(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"


def test_compute_abs_diff_mean_anglez(
    sleep_detection: analytics.GgirSleepDetection,
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


def test_spt_window(sleep_detection: analytics.GgirSleepDetection) -> None:
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


def test_spt_window_null(sleep_detection: analytics.GgirSleepDetection) -> None:
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


def test_calculate_sib_periods(sleep_detection: analytics.GgirSleepDetection) -> None:
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


def test_find_onset_wakeup_times(sleep_detection: analytics.GgirSleepDetection) -> None:
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


def test_run_sleep_detection(sleep_detection: analytics.GgirSleepDetection) -> None:
    """Test the full sleep detection process."""
    result = sleep_detection.run_sleep_detection()

    assert result == []
    assert isinstance(result, List)


def test_physical_activity_thresholds() -> None:
    """Test the physical_activity_thresholds method."""
    tmp_data = np.arange(4)
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + datetime.timedelta(seconds=i) for i in range(4)]
    time = pl.Series("time", dummy_datetime_list)
    tmp_measurement = models.Measurement(measurements=tmp_data, time=time)
    thresholds = (0, 1, 2)
    expected_result = np.array(["inactive", "light", "moderate", "vigorous"])

    result = analytics.compute_physical_activty_categories(tmp_measurement, thresholds)

    assert np.array_equal(result.measurements, expected_result)
    assert np.array_equal(result.time, time)


def test_bad_physical_activity_thresholds() -> None:
    """Test bad values being passed to physical_activity_thresholds."""
    tmp_data = np.arange(4)
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + datetime.timedelta(seconds=i) for i in range(4)]
    time = pl.Series("time", dummy_datetime_list)
    tmp_measurement = models.Measurement(measurements=tmp_data, time=time)

    with pytest.raises(
        ValueError,
        match="Thresholds must be positive, unique, and given in ascending order.",
    ):
        analytics.compute_physical_activty_categories(
            tmp_measurement, thresholds=(10, 5, 1)
        )


def test_sleep_cleanup() -> None:
    """Test the sleep_cleanup method."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_datetime_list = [
        dummy_date + datetime.timedelta(seconds=i) for i in range(3600)
    ]
    time = pl.Series("time", dummy_datetime_list)
    nonwear_data = np.zeros(3600)
    nonwear_data[2000:2400] = 1
    nonwear_measurement = models.Measurement(measurements=nonwear_data, time=time)
    sleep_windows = [
        analytics.SleepWindow(
            onset=dummy_date + datetime.timedelta(minutes=10),
            wakeup=dummy_date + datetime.timedelta(minutes=40),
        )
    ]

    expected_result = np.zeros(len(nonwear_measurement.time))
    expected_result[600:2000] = 1

    result = analytics.sleep_cleanup(
        sleep_windows=sleep_windows, nonwear_measurement=nonwear_measurement
    )

    assert len(result.time) == 3600
    assert np.array_equal(result.measurements, expected_result)
