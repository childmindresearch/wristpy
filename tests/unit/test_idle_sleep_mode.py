"""Testing the idle_sleep_mode functions."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from wristpy.core import models
from wristpy.processing import idle_sleep_mode_imputation


@pytest.mark.parametrize(
    "sampling_rate, effective_sampling_rate", [(30, 25), (20, 20), (1, 1)]
)
def test_idle_sleep_mode_resampling(
    sampling_rate: int, effective_sampling_rate: int
) -> None:
    """Test the idle_sleep_mode function."""
    num_samples = 10000
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [
        dummy_date + timedelta(seconds=i / sampling_rate) for i in range(num_samples)
    ]
    test_time = pl.Series("time", dummy_datetime_list, dtype=pl.Datetime("ns"))
    acceleration = models.Measurement(
        measurements=np.ones((num_samples, 3)), time=test_time
    )

    filled_acceleration = idle_sleep_mode_imputation.impute_idle_sleep_mode_gaps(
        acceleration
    )

    assert (
        np.mean(
            filled_acceleration.time.diff()
            .drop_nulls()
            .dt.total_nanoseconds()
            .to_numpy()
            .astype(dtype=float)
        )
        == 1e9 / effective_sampling_rate
    )


def test_idle_sleep_mode_gap_fill() -> None:
    """Test the idle_sleep_mode gap fill functionality."""
    num_samples = 10000
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [
        dummy_date + timedelta(seconds=i) for i in range(num_samples // 2)
    ]
    time_gap = dummy_date + timedelta(seconds=(1000))
    dummy_datetime_list += [
        time_gap + timedelta(seconds=i) for i in range(num_samples // 2, num_samples)
    ]
    test_time = pl.Series("time", dummy_datetime_list, dtype=pl.Datetime("ns"))
    acceleration = models.Measurement(
        measurements=np.ones((num_samples, 3)), time=test_time
    )
    expected_acceleration = (np.finfo(float).eps, np.finfo(float).eps, -1)

    filled_acceleration = idle_sleep_mode_imputation.impute_idle_sleep_mode_gaps(
        acceleration
    )

    assert len(filled_acceleration.measurements) > len(acceleration.measurements)
    assert (
        np.mean(
            filled_acceleration.time.diff()
            .drop_nulls()
            .dt.total_nanoseconds()
            .to_numpy()
            .astype(dtype=float)
        )
        == 1e9
    )
    assert np.all(filled_acceleration.measurements[5010] == expected_acceleration)
