"""Generate and filter sleep windows from predicted onsets and offsets."""

import numpy as np
import polars as pl

from wristpy.core import models


def generate_sleep_windows(
    onset_scores: models.Measurement,
    offset_scores: models.Measurement,
    min_duration: float = 900.0,
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
    onset_times = onset_scores.time.to_numpy().astype("datetime64[s]")
    offset_times = offset_scores.time.to_numpy().astype("datetime64[s]")

    durations = offset_times[:, None] - onset_times[None, :]
    duration_mask = durations > min_duration
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

    def test_window(window: tuple[pl.Datetime, pl.Datetime, float]) -> bool:
        window_score = (
            sleep_scores_df.filter(pl.col("time").is_between(window[0], window[1]))
            .select("score")
            .min()
        )
        return window_score > threshold

    filtered = windows.filter(windows.map_rows(test_window))
    return filtered


def nms_windows(windows: pl.DataFrame, iou_threshold: float = 0.75) -> pl.DataFrame:
    """De-duplicate sleep windows by non-max suppression with given IoU threshold."""
    windows = windows.sort("score", descending=True)

    bounds = windows.select("onset", "offset").to_numpy()
    mask = np.zeros(len(bounds), dtype="bool")
    mask[0] = True

    for ii in range(1, len(bounds)):
        iou = pairwise_iou(bounds[mask], bounds[ii : ii + 1])
        if iou.max() < iou_threshold:
            mask[ii] = True

    filtered = windows.filter(mask)
    return filtered


def pairwise_iou(bounds1: np.ndarray, bounds2: np.ndarray) -> np.ndarray:
    """Compute the IoU between all pairs of 1d bounding boxes."""
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
    durs = np.minimum(r1[:, None], r2[None, :]) - np.maximum(l1[:, None], l2[None, :])
    durs = np.maximum(durs, 0)
    return durs
