"""Test the function of the nonwear_utils module."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from wristpy.core import models
from wristpy.processing import processing_utils

_SLEEP_START = datetime(2024, 12, 8)


def _sleep_measurement(seconds: list[int], values: list[bool]) -> models.Measurement:
    """Create a small sleep-bout measurement with offsets from a fixed start."""
    time = pl.Series(
        "time",
        [_SLEEP_START + timedelta(seconds=second) for second in seconds],
        dtype=pl.Datetime("ns"),
    )
    return models.Measurement(measurements=np.asarray(values), time=time)


def test_nonwear_majority_vote() -> None:
    """Tests the majority vote function for nonwear."""
    time1 = [
        datetime(1990, 1, 1, 1, 1) + timedelta(seconds=60 * idx) for idx in range(900)
    ]
    time3 = [
        datetime(1990, 1, 1, 1, 1, 10) + timedelta(seconds=100 * idx)
        for idx in range(11)
    ]
    nonwear1 = models.Measurement(
        measurements=np.ones(len(time1)),
        time=pl.Series("time", time1, dtype=pl.Datetime("ns")),
    )
    nonwear2 = models.Measurement(
        measurements=np.ones(len(time1)),
        time=pl.Series("time", time1, dtype=pl.Datetime("ns")),
    )
    nonwear3 = models.Measurement(
        measurements=np.ones(len(time3)),
        time=pl.Series("time", time3, dtype=pl.Datetime("ns")),
    )

    nonwear = processing_utils._majority_vote_non_wear(
        [nonwear1, nonwear2, nonwear3],
        temporal_resolution=5,
    )

    assert np.all(nonwear.measurements == 1)


def test_nonwear_majority_vote_even() -> None:
    """Tests the majority vote function for nonwear, with even input."""
    time1 = [
        datetime(1990, 1, 1, 1, 1) + timedelta(seconds=60 * idx) for idx in range(900)
    ]
    nonwear1 = models.Measurement(
        measurements=np.ones(len(time1)),
        time=pl.Series("time", time1, dtype=pl.Datetime("ns")),
    )
    nonwear2 = models.Measurement(
        measurements=np.zeros(len(time1)),
        time=pl.Series("time", time1, dtype=pl.Datetime("ns")),
    )

    nonwear = processing_utils._majority_vote_non_wear(
        [nonwear1, nonwear2],
        temporal_resolution=5,
    )

    assert np.all(nonwear.measurements == 0)


def test_nonwear_majority_vote_empty() -> None:
    """Tests the majority vote function for nonwear when no inputs provided."""
    with pytest.raises(ValueError):
        processing_utils._majority_vote_non_wear(
            nonwear_measurements=[],
            temporal_resolution=5,
        )


def test_nonwear_dispatcher_default() -> None:
    """Tests the nonwear dispatcher function."""
    num_samples = int(1e5)
    time_list = [
        datetime(2024, 5, 2) + timedelta(milliseconds=100 * i)
        for i in range(num_samples)
    ]
    time = pl.Series("time", time_list, dtype=pl.Datetime("ns"))
    acceleration = models.Measurement(measurements=np.ones((num_samples, 3)), time=time)
    temperature = models.Measurement(
        measurements=np.random.randint(low=22, high=32, size=num_samples), time=time
    )

    nonwear_array = processing_utils.get_nonwear_measurements(acceleration, temperature)

    assert isinstance(nonwear_array, models.Measurement)


def test_nonwear_dispatcher_multiple() -> None:
    """Tests nonwear dispatcher with multiple nonwear algorithms requested."""
    num_samples = int(1e5)
    time_list = [
        datetime(2024, 5, 2) + timedelta(milliseconds=100 * i)
        for i in range(num_samples)
    ]
    time = pl.Series("time", time_list, dtype=pl.Datetime("ns"))
    acceleration = models.Measurement(measurements=np.ones((num_samples, 3)), time=time)
    temperature = models.Measurement(
        measurements=np.random.randint(low=22, high=32, size=num_samples), time=time
    )

    nonwear_array = processing_utils.get_nonwear_measurements(
        acceleration, temperature, ["cta", "detach", "ggir"]
    )

    assert isinstance(nonwear_array, models.Measurement)


def test_nonwear_dispatcher_unknown_algo() -> None:
    """Tests nonwear dispatcher with incorrect algorithm."""
    num_samples = int(1e5)
    time_list = [
        datetime(2024, 5, 2) + timedelta(milliseconds=100 * i)
        for i in range(num_samples)
    ]
    time = pl.Series("time", time_list, dtype=pl.Datetime("ns"))
    acceleration = models.Measurement(measurements=np.ones((num_samples, 3)), time=time)
    temperature = models.Measurement(
        measurements=np.random.randint(low=22, high=32, size=num_samples), time=time
    )

    with pytest.raises(ValueError):
        processing_utils.get_nonwear_measurements(
            acceleration,
            temperature,
            ["unknown"],  # type: ignore [list-item] #Violating Literal to raise ValueError
        )


def test_nonwear_dispatcher_missing_temperature() -> None:
    """Tests nonwear dispatcher with missing temperature."""
    num_samples = int(1e5)
    time_list = [
        datetime(2024, 5, 2) + timedelta(milliseconds=100 * i)
        for i in range(num_samples)
    ]
    time = pl.Series("time", time_list, dtype=pl.Datetime("ns"))
    acceleration = models.Measurement(measurements=np.ones((num_samples, 3)), time=time)

    with pytest.raises(ValueError):
        processing_utils.get_nonwear_measurements(
            acceleration, non_wear_algorithms=["cta"]
        )


@pytest.mark.parametrize(
    (
        "data_seconds",
        "data_values",
        "reference_seconds",
        "reference_values",
        "interval",
        "expected_result",
    ),
    [
        pytest.param(
            [0, 3, 6],
            [False, True, False],
            [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            [False] * 11,
            1,
            [
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
            ],
            id="upsample-extrapolate",
        ),
        pytest.param(
            [0, 3, 6],
            [False, True, False],
            [2, 3, 4],
            [False] * 3,
            1,
            [True, True, True],
            id="upsample-truncate",
        ),
        pytest.param(
            [0, 1, 2, 3, 4, 5, 6],
            [False, False, False, True, True, True, False],
            [0, 3, 6],
            [False] * 3,
            3,
            [False, True, False],
            id="downsample",
        ),
    ],
)
def test_synchronize_measurements(
    data_seconds: list[int],
    data_values: list[bool],
    reference_seconds: list[int],
    reference_values: list[bool],
    interval: int,
    expected_result: list[bool],
) -> None:
    """Test synchronization of measurements across different test cases."""
    data = _sleep_measurement(
        seconds=data_seconds,
        values=data_values,
    )
    reference = _sleep_measurement(
        seconds=reference_seconds,
        values=reference_values,
    )

    result = processing_utils.synchronize_measurements(
        data,
        reference,
        interval,
    )

    assert np.array_equal(
        result.measurements,
        np.asarray(expected_result, dtype=bool),
    )
    assert result.time.equals(reference.time)
    assert result.measurements.dtype == np.bool_


@pytest.mark.parametrize(
    (
        "data_seconds",
        "data_values",
        "reference_seconds",
        "reference_values",
        "expected_result",
    ),
    [
        pytest.param(
            [0, 10, 20],
            [False, True, False],
            [-5, 0, 5, 10, 15, 20, 25],
            [False] * 7,
            [
                False,
                False,
                False,
                True,
                True,
                False,
                False,
            ],
            id="upsample-extrapolate",
        ),
        pytest.param(
            [0, 10, 20],
            [False, True, False],
            [5, 10, 15],
            [False] * 3,
            [False, True, True],
            id="upsample-truncate",
        ),
        pytest.param(
            [0, 5, 10, 15, 20],
            [False, True, True, False, True],
            [0, 10, 20],
            [False] * 3,
            [False, True, True],
            id="downsample",
        ),
    ],
)
def test_synchronize_sleep_bouts(
    data_seconds: list[int],
    data_values: list[bool],
    reference_seconds: list[int],
    reference_values: list[bool],
    expected_result: list[bool],
) -> None:
    """Test synchronization of sleep bouts across different test cases."""
    data = _sleep_measurement(
        seconds=data_seconds,
        values=data_values,
    )
    reference = _sleep_measurement(
        seconds=reference_seconds,
        values=reference_values,
    )

    result = processing_utils._synchronize_sleep_bouts(data, reference)

    assert np.array_equal(
        result.measurements,
        np.asarray(expected_result, dtype=bool),
    )
    assert result.time.equals(reference.time)
    assert result.measurements.dtype == np.bool_
