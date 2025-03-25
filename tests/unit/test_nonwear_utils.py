"""Test the function of the nonwear_utils module."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from wristpy.core import models
from wristpy.processing import nonwear_utils


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

    nonwear = nonwear_utils.majority_vote_non_wear(
        nonwear1,
        nonwear2,
        nonwear3,
        temporal_resolution=5,
    )

    assert np.all(nonwear.measurements == 1)


def test_nonwear_majority_vote_even() -> None:
    """Tests the majority vote function for nonwear."""
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

    nonwear = nonwear_utils.majority_vote_non_wear(
        nonwear1,
        nonwear2,
        temporal_resolution=5,
    )

    assert np.all(nonwear.measurements == 0)
