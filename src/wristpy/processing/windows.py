"""Generate and filter sleep windows from predicted onsets and offsets."""

from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import polars as pl

from wristpy.core import models


def generate_sleep_windows(
    onset_scores: models.Measurement,
    offset_scores: models.Measurement,
    min_duration: int = 900,
) -> pl.DataFrame:
    """Generate sleep windows from predicted onsets and wakeups.

    Args:
        onset_scores: onset event prediction scores (non-negative)
        offset_scores: offset event prediction scores (non-negative)
        min_duration: minimum duration of a sleep window in seconds

    Returns:
        A dataframe with columns 'onset', 'wakeup', and 'score', which is the geometric
        mean of onset and wakeup scores.
    """
    onset_times = onset_scores.time.to_numpy()
    offset_times = offset_scores.time.to_numpy()

    durations = offset_times[:, None] - onset_times[None, :]
    duration_mask = durations > np.timedelta64(min_duration, "s")
    offset_ind, onset_ind = duration_mask.nonzero()
    scores = np.sqrt(
        onset_scores.measurements[onset_ind] * offset_scores.measurements[offset_ind]
    )

    windows = pl.DataFrame(
        {
            "onset": onset_times[onset_ind],
            "offset": offset_times[offset_ind],
            "score": scores,
        }
    )
    return windows


def filter_non_sleep_windows(
    windows: pl.DataFrame, sleep_scores: models.Measurement, threshold: float = 0.5
) -> pl.DataFrame:
    """Exclude sleep windows whose sleep scores are not all above a threshold.

    Args:
        windows: sleep windows dataframe with columns 'onset', 'wakeup', 'score'
        sleep_scores: instantaneous sleep score measurement
        threshold: minimum sleep score threshold

    Returns:
        filtered windows dataframe
    """
    sleep_scores_df = pl.DataFrame(
        {"time": sleep_scores.time, "score": sleep_scores.measurements}
    )

    def test_window(window: tuple[datetime, datetime, float]) -> bool:
        window_score = (
            sleep_scores_df.filter(pl.col("time").is_between(window[0], window[1]))
            .select("score")
            .min()
            .item()
        )
        return window_score > threshold

    mask = windows.map_rows(test_window)
    filtered = windows.filter(mask[:, 0])
    return filtered


def nms_windows(
    windows: pl.DataFrame,
    metric: Literal["iou", "overlap"] = "overlap",
    threshold: float = 0.5,
) -> pl.DataFrame:
    """De-duplicate sleep windows by non-max suppression with given threshold."""
    windows = windows.sort("score", descending=True)

    bounds = windows_to_bounds(windows)
    mask = np.zeros(len(bounds), dtype="bool")
    mask[0] = True

    for ii in range(1, len(bounds)):
        if metric == "iou":
            score = pairwise_iou(bounds[mask], bounds[ii : ii + 1])
        else:
            score = pairwise_overlap(bounds[mask], bounds[ii : ii + 1])

        if score.max() < threshold:
            mask[ii] = True

    filtered = windows.filter(mask)
    return filtered


def windows_to_bounds(
    windows: pl.DataFrame,
    unit: Literal["s", "ms", "us", "ns"] = "ms",
) -> np.ndarray:
    """Extract interval bounds from windows dataframe."""
    bounds = windows.select("onset", "offset").to_numpy()

    # convert from datetime to float
    bounds = (bounds - bounds.min()).astype(f"timedelta64[{unit}]")
    bounds = bounds.astype("float64")
    return bounds


def pairwise_overlap(bounds1: np.ndarray, bounds2: np.ndarray) -> np.ndarray:
    """Compute the overlap between all pairs of 1d bounding boxes."""
    bounds1 = np.atleast_2d(bounds1)
    bounds2 = np.atleast_2d(bounds2)

    l1, r1 = bounds1.T
    l2, r2 = bounds2.T
    durs1 = np.maximum(r1 - l1, 0.0)
    durs2 = np.maximum(r2 - l2, 0.0)
    intersection = pairwise_intersection(bounds1, bounds2)
    min_durs = np.minimum(durs1[:, None], durs2[None, :])
    overlap = np.where(min_durs > 0, intersection / min_durs, 0.0)
    return overlap


def pairwise_iou(bounds1: np.ndarray, bounds2: np.ndarray) -> np.ndarray:
    """Compute the IoU between all pairs of 1d bounding boxes."""
    bounds1 = np.atleast_2d(bounds1)
    bounds2 = np.atleast_2d(bounds2)

    l1, r1 = bounds1.T
    l2, r2 = bounds2.T
    durs1 = np.maximum(r1 - l1, 0.0)
    durs2 = np.maximum(r2 - l2, 0.0)
    intersection = pairwise_intersection(bounds1, bounds2)
    union = durs1[:, None] + durs2[None, :] - intersection
    iou = np.where(union > 0, intersection / union, 0.0)
    return iou


def pairwise_intersection(bounds1: np.ndarray, bounds2: np.ndarray) -> np.ndarray:
    """Compute the intersection between all pairs of 1d bounding boxes."""
    l1, r1 = bounds1.T
    l2, r2 = bounds2.T
    lint = np.maximum(l1[:, None], l2[None, :])
    rint = np.minimum(r1[:, None], r2[None, :])
    intersection = rint - lint
    intersection = np.maximum(intersection, 0)
    return intersection


def find_segments(
    sleep_scores: models.Measurement,
    duration: timedelta = timedelta(days=1.0),
    threshold: float = 0.1,
) -> np.ndarray:
    """Find non-sleep time points where the long timeseries can be split into segments.

    Args:
        sleep_scores: instantaneous sleep score measurement
        duration: approximate length of the segment
        threshold: score threshold for non-sleep time points

    Returns:
        array of timestamps indicating the segment cut points.
    """
    times = sleep_scores.time.to_numpy()
    score_values = sleep_scores.measurements
    duration = np.timedelta64(duration)
    assert np.all(np.diff(times) > np.timedelta64(0, "s")), "timeseries is not sorted"

    cuts = []
    start = times[0]

    while start + duration < times[-1]:
        # search for a non-sleep point where we can cut the timeseries
        # find open and close points of the search window
        open_idx = np.searchsorted(times, start + duration, side="right")
        close_idx = np.searchsorted(times, times[open_idx] + duration, side="left")

        # find the first non-sleep point in the search window
        search_scores = score_values[open_idx:close_idx]
        offset = np.argmax(search_scores < threshold)
        start = times[open_idx + offset]

        if search_scores[offset] < threshold:
            cuts.append(start)

    cuts = np.array(cuts)
    return cuts
