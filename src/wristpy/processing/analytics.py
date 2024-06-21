"""Calculate sleep onset and wake times."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np
import polars as pl

from wristpy.core import computations, models


@dataclass
class SleepWindow:
    """Dataclass to store sleep window information.

    Attributes:
        onset: the time the participant fell asleep.
        wakeup: the time the participant woke up
    """

    onset: List[datetime]
    wakeup: List[datetime]


class SleepDetection:
    """Class to detect sleep onset and wake times."""

    def __init__(
        self, anglez: models.Measurement, nonwear: models.Measurement, method: str
    ) -> None:
        """Initialize the SleepDetection class."""
        self.anglez = anglez
        self.nonwear = nonwear
        self.method = method

    def run(self, method: str) -> SleepWindow:
        """Run the sleep detection algorithm."""
        if method == "GGIR":
            spt_periods = self._spt_window()
            sib_periods = self._calculate_sib_periods()
            sleep_onset_wakeup = self._find_onset_wakeup_times(spt_periods, sib_periods)

        return sleep_onset_wakeup

    def _spt_window(
        self, anglez_data: models.Measurement, threshold: float = 0.13
    ) -> models.Measurement:
        """Implement GGIR Heuristic distribution of z angle.

        This function finds the absolute difference of the anglez data over 5s windows.
        We then find when the 5-minute rolling median of that difference is below a
        threshold (chosen middle point of GGIR range).
        We then find blocks of 30 minutes (360 blocks of 5s) where the rolling median is
        below the threshold.
        We then fill gaps that are less than 60 minutes.

        Args:
            anglez_data: the raw anglez data, calculated from calibrated acceleration.
            threshold: the threshold for the distribution of z angle.

        Returns:
            A Measurement instance with values set to 1 indicating identified
            SPT windows and corresponding time stamps.
        """
        anglez_abs_diff = self._compute_abs_diff_mean_anglez(anglez_data)
        anglez_median_long_epoch = computations.moving_median(anglez_abs_diff, 300)
        below_threshold = (anglez_median_long_epoch.measurements < threshold).astype(
            int
        )

        block_length = 360
        kernel = np.ones(block_length, dtype=int)
        convolved = np.convolve(below_threshold, kernel, mode="same")
        long_blocks_idx = np.where(convolved == block_length)[0]
        sleep_idx_array = np.zeros(len(below_threshold))
        sleep_idx_array[long_blocks_idx] = 1

        zeros_idx = np.where(sleep_idx_array == 0)[0]
        n_blocks = np.split(zeros_idx, np.where(np.diff(zeros_idx) != 1)[0] + 1)
        for block_idx in n_blocks:
            if len(block_idx) < (block_length * 2):
                sleep_idx_array[block_idx] = 1

        return models.Measurement(
            measurements=sleep_idx_array, time=anglez_median_long_epoch.time
        )

    def _calculate_sib_periods(
        self, anglez_data: models.Measurement, threshold_degrees: int = 5
    ) -> models.Measurement:
        """Find the sustained inactivity bouts.

        This function finds the absolute dtifference of the anglez data over 5s windows.
        We then find the 5-minute windows where all of the differences are below a
        threshold (defaults to 5 degrees).

        Args:
            anglez_data: the raw anglez data, calculated from calibrated acceleration.
            threshold_degrees: the threshold, in degrees, for inactivity.

        Returns:
            A Measurement instance with values set to 1 indicating identified SIB
            windows, and corresponding time stamps.
        """
        anglez_abs_diff = self._compute_abs_diff_mean_anglez(anglez_data)

        anglez_pl_df = pl.DataFrame(
            {"time": anglez_abs_diff.time, "angz_diff": anglez_abs_diff.measurements}
        )
        anglez_pl_df = anglez_pl_df.with_columns(pl.col("time").set_sorted())
        anglez_grouped_by_window_length = anglez_pl_df.group_by_dynamic(
            index_column="time", every="5m"
        ).agg([pl.all().exclude(["time"])])
        flag = anglez_grouped_by_window_length["angz_diff"].map_elements(
            lambda lst: all(x < threshold_degrees for x in lst)
        )

        return models.Measurement(
            measurements=(flag.to_numpy()).astype(int),
            time=anglez_grouped_by_window_length["time"],
        )

    def _compute_abs_diff_mean_anglez(
        self, anglez_data: models.Measurement, window_size_seconds: int = 5
    ) -> models.Measurement:
        """Helper function to compute the absolute difference of averaged anglez data.

        Args:
            anglez_data: the raw anglez data.
                ##Note if we have access to the epoch1 anglez data this function will be
                modified or removed
            window_size_seconds: the window size in seconds to average the anglez data.

        Returns:
            A Measurement instance with the absolute difference of the anglez data.
        """
        anglez_epoch1 = computations.moving_mean(anglez_data, window_size_seconds)
        absolute_diff = np.abs(np.diff(anglez_epoch1.measurements))

        return models.Measurement(measurements=absolute_diff, time=anglez_data.time[1:])

    def _find_onset_wakeup_times(
        self, spt_periods: models.Measurement, sib_periods: models.Measurement
    ) -> SleepWindow:
        """Find the sleep onset and wake up times.

        This function is implemented as follows.
        First, we use the spt_window as a guider for potential sleep.
        Then we find overlapping sustained inactivity bouts (sib_periods == 1).
        Sleep onset is defined as the start of the first sib_period that overlaps
        with a specific spt_window.
        Sleep wakeup is the end of the last sib that overlaps with a spt window.

        Args:
            spt_periods: the sleep period guider windows.
            sib_periods: the sustained inactivity bouts.

        Returns:
            A SleepWindow instance with sleep onset and wake up times.
            If there is no overlap between spt_windows and sib_periods,
            the onset and wakeup lists will be empty.
        """
        onset = []
        wakeup = []
        for spt in spt_periods:
            min_onset = None
            max_wakeup = None
            for sib in sib_periods:
                if sib[0] <= spt[1] and sib[1] >= spt[0]:
                    if min_onset is None or sib[0] < min_onset:
                        min_onset = sib[0]
                    if max_wakeup is None or sib[1] > max_wakeup:
                        max_wakeup = sib[1]
            if min_onset is not None and max_wakeup is not None:
                onset.append(min_onset)
                wakeup.append(max_wakeup)

        return SleepWindow(onset=onset, wakeup=wakeup)

    def _find_periods(
        self, window_measurement: models.Measurement
    ) -> List[Tuple[datetime, datetime]]:
        """Find periods where window_measurement is equal to 1.

        This is intended to be a helper function for the _find_onset_wakeup_times to
        find periods where either the spt_window or sib_periods are equal to 1.

        Args:
            window_measurement: the Measurement instance, intended to be
                either the spt_window or sib_period.

        Returns:
            A list of tuples, where each tuple contains the start and end times of
            a period.
        """
        periods = []
        start_time = None

        for time, value in zip(
            window_measurement.time, window_measurement.measurements
        ):
            if value == 1 and start_time is None:
                start_time = time
            elif value != 1 and start_time is not None:
                periods.append((start_time, time))
                start_time = None

        if start_time is not None:
            periods.append((start_time, window_measurement.time[-1]))

        return periods
