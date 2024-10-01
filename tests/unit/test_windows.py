"""Tests for sleep window generation and filtering."""

import datetime
import math

import numpy as np
import polars as pl
import pytest

from wristpy.core import models
from wristpy.processing import windows as win


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Shared numpy random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def onsets_offsets(
    rng: np.random.Generator,
) -> tuple[models.Measurement, models.Measurement]:
    """Random onset and offset event measurements."""
    time = pl.Series(
        [
            datetime.datetime(2024, 10, 1) + datetime.timedelta(seconds=secs)
            for secs in range(0, 12000, 100)
        ]
    )

    onset_score_values = rng.random(len(time))
    offset_score_values = rng.random(len(time))
    onset_scores = models.Measurement(measurements=onset_score_values, time=time)
    offset_scores = models.Measurement(measurements=offset_score_values, time=time)
    return onset_scores, offset_scores


@pytest.fixture(scope="module")
def windows(
    rng: np.random.Generator,
    onsets_offsets: tuple[models.Measurement, models.Measurement],
) -> pl.DataFrame:
    """Random sleep windows dataframe."""
    onset_scores, _ = onsets_offsets
    onset_times = onset_scores.time.to_numpy()

    durations = rng.uniform(500, 1000, size=len(onset_times)).astype("timedelta64[s]")
    offset_times = onset_times + durations

    windows = pl.DataFrame(
        {
            "onset": onset_times,
            "offset": offset_times,
            "score": onset_scores.measurements,
        }
    )
    return windows


@pytest.fixture(scope="module")
def sleep_scores(
    onsets_offsets: tuple[models.Measurement, models.Measurement],
) -> models.Measurement:
    """Contrived intantaneous sleep scores of all ones with middle third set to 0."""
    onset_scores, _ = onsets_offsets
    n_timesteps = len(onset_scores.measurements)

    scores = np.concatenate(
        [
            np.ones(n_timesteps // 3),
            np.zeros(n_timesteps // 3),
            np.ones(n_timesteps // 3),
        ]
    )

    sleep_scores = models.Measurement(measurements=scores, time=onset_scores.time)
    return sleep_scores


@pytest.fixture(scope="module")
def bounds_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random pair of left/right bounds arrays."""
    l1, w1 = rng.random((2, 10))
    l2, w2 = rng.random((2, 20))
    bounds1 = np.stack([l1, l1 + w1], axis=1)
    bounds2 = np.stack([l2, l2 + w2], axis=1)
    return bounds1, bounds2


def test_generate_sleep_windows(
    onsets_offsets: tuple[models.Measurement, models.Measurement],
) -> None:
    """Test generation of all sleep windows from onset/offset events."""
    onset_scores, offset_scores = onsets_offsets
    windows = win.generate_sleep_windows(onset_scores, offset_scores, min_duration=60)

    assert windows.columns == ["onset", "offset", "score"]
    assert len(windows) == 7140

    assert windows["score"].is_between(0, 1).all()

    durations = (windows["offset"] - windows["onset"]).to_numpy()
    assert (durations > np.timedelta64(60, "s")).all()


def test_filter_non_sleep_windows(
    windows: pl.DataFrame, sleep_scores: models.Measurement
) -> None:
    """Test filtering out of sleep windows that include clear non-sleep periods."""
    filtered = win.filter_non_sleep_windows(windows, sleep_scores)

    # since the sleep scores have middle third set to zero, expect middle third of
    # windows to be removed.
    assert math.isclose(len(filtered) / len(windows), 2 / 3, rel_tol=0.1, abs_tol=0.1)


def test_nms_windows(windows: pl.DataFrame) -> None:
    """Test non-max suppression of overlapping windows."""
    filtered = win.nms_windows(windows, iou_threshold=0.75)

    assert len(filtered) == 88

    # check that all pairwise IoUs are below threshold
    bounds = win.windows_to_bounds(filtered)
    iou = win.pairwise_iou(bounds, bounds)
    rind, cind = np.triu_indices(len(bounds), 1)
    assert np.max(iou[rind, cind]) < 0.75


def test_pairwise_iou(bounds_pair: tuple[np.ndarray, np.ndarray]) -> None:
    """Test pairwise IoU."""
    bounds1, bounds2 = bounds_pair
    iou = win.pairwise_iou(bounds1, bounds2)
    assert iou.shape == (len(bounds1), len(bounds2))
    assert np.all(iou >= 0)
    assert np.all(iou <= 1)


def test_pairwise_intersection(bounds_pair: tuple[np.ndarray, np.ndarray]) -> None:
    """Test pairwise intersection."""
    bounds1, bounds2 = bounds_pair
    intersection = win.pairwise_intersection(bounds1, bounds2)
    assert intersection.shape == (len(bounds1), len(bounds2))
    assert np.all(intersection >= 0)
    assert np.all(intersection <= 1)
