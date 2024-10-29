"""Calculate sleep onset and wake up times."""

import abc
import datetime
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import polars as pl

from wristpy.core import computations, config, models

logger = config.get_logger()


@dataclass
class SleepWindow:
    """Dataclass to store sleep window information.

    Attributes:
        onset: the predicted start time of the sleep window.
        wakeup: the predicted end time of the sleep window.
    """

    onset: Union[datetime.datetime, List]
    wakeup: Union[datetime.datetime, List]


class AbstractSleepDetector(abc.ABC):
    """Abstract class defining the interface for sleep detection algorithms."""

    @abc.abstractmethod
    def __init__(self, anglez: models.Measurement) -> None:
        """Initialization function for the sleep detection algorithm.

        Must contain the anglez data as an input.
        """
        pass

    @abc.abstractmethod
    def run_sleep_detection(self) -> List[SleepWindow]:
        """Sleep Detector must contain a run_sleep_detection function.

        The function must return a list of SleepWindow objects.
        """
        pass


class GgirSleepDetection(AbstractSleepDetector):
    """Sleep Detection algorithm based on the GGIR method.

    This class implements the GGIR method for sleep detection. The method uses the
    angle-z data to find periods of sleep. It returns a list of SleepWindow objects.

    Attributes:
        anglez (models.Measurement): The angle-z data, calculated from the calibrated
            accelerometer data.
    """

    def __init__(
        self,
        anglez: models.Measurement,
    ) -> None:
        """Initialize the SleepDetection class with default attributes.

        Args:
            anglez: the raw anglez data, calculated from calibrated acceleration.
        """
        self.anglez = anglez

    def run_sleep_detection(self) -> List[SleepWindow]:
        """Run the GGIR sleep detection.

        This algorithm uses the angle-z data to first find potential sleep periods
        (SPT windows) and then find sustained inactivity bouts (SIB periods). The
        sleep onset and wake up times are then calculated based on the overlap between
        the SPT windows and SIB periods.

        Returns:
            A list of SleepWindow instances, each instance contains a sleep onset/wakeup
            time pair.
        """
        logger.debug("Beginning sleep detection.")
        spt_window = self._spt_window(self.anglez)
        sib_periods = self._calculate_sib_periods(self.anglez)
        spt_window_periods = _find_periods(spt_window)
        sib_window_periods = _find_periods(sib_periods)
        sleep_onset_wakeup = self._find_onset_wakeup_times(
            spt_window_periods, sib_window_periods
        )
        logger.debug(
            "Sleep detection complete. Windows detected: %s", len(sleep_onset_wakeup)
        )
        return sleep_onset_wakeup

    def _spt_window(
        self, anglez_data: models.Measurement, threshold: float = 0.2
    ) -> models.Measurement:
        """Implement Heuristic distribution of z angle (HDCZA) for SPT window detection.

        This function finds the absolute difference of the anglez data over 5s windows.
        We find the 5-minute rolling median of that difference.
        Next, we find when that 5-minute median is above a specified threshold, taken as
        the new default value from the GGIR implementation of the HDCZA algorithm. This
        represents non-sleep candidates. The logical not of this is the sleep candidate.
        We then find long blocks (30 minutes) when the threshold criteria is met.
        Any gaps in SPT windows that are less than 60 minutes are filled.

        Args:
            anglez_data: the raw anglez data, calculated from calibrated acceleration.
            threshold: the threshold for the distribution of z angle, chosen as
                0.2 based on new GGIR default.

        Returns:
            A Measurement instance with values set to 1 indicating identified
            SPT windows and corresponding time stamps.

        References:
            van Hees, V.T., et al. Estimating sleep parameters using an
                accelerometer without sleep diary. Sci Rep 8, 12975 (2018).
                https://doi.org/10.1038/s41598-018-31266-z
        """
        logger.debug("Finding spt windows, Threshold: %s", threshold)
        long_epoch_median = 300
        long_block = 360
        short_block_gap = 720

        anglez_abs_diff = self._compute_abs_diff_mean_anglez(anglez_data)
        anglez_median_long_epoch = computations.moving_median(
            anglez_abs_diff, long_epoch_median
        )
        non_sleep_candidates = (
            anglez_median_long_epoch.measurements >= threshold
        ).flatten()

        sleep_candidates = np.logical_not(
            self._fill_false_blocks(non_sleep_candidates, long_block)
        )

        sleep_idx_array_filled = self._fill_false_blocks(
            sleep_candidates, short_block_gap
        )
        return models.Measurement(
            measurements=sleep_idx_array_filled, time=anglez_median_long_epoch.time
        )

    def _calculate_sib_periods(
        self, anglez_data: models.Measurement, threshold_degrees: int = 5
    ) -> models.Measurement:
        """Find the sustained inactivity bouts.

        This function finds the absolute difference of the anglez data over 5s windows.
        We then find the 5-minute windows where all of the differences are below a
        threshold (defaults to 5 degrees).

        Args:
            anglez_data: the raw anglez data, calculated from calibrated acceleration.
            threshold_degrees: the threshold, in degrees, for inactivity.

        Returns:
            A Measurement instance with values set to 1 indicating identified SIB
            windows, and corresponding time stamps.

        References:
            van Hees, V. T. et al. A Novel, Open Access Method to Assess Sleep
                Duration Using a Wrist-Worn Accelerometer. PLoS One 10, e0142533 (2015).
                https://doi.org/10.1371/journal.pone.0142533
        """
        logger.debug("Calculating SIB period threshold: %s degrees", threshold_degrees)
        anglez_abs_diff = self._compute_abs_diff_mean_anglez(anglez_data)

        anglez_pl_df = pl.DataFrame(
            {"time": anglez_abs_diff.time, "angz_diff": anglez_abs_diff.measurements}
        )

        anglez_pl_df = anglez_pl_df.with_columns(pl.col("time").set_sorted())
        anglez_grouped_by_window_length = anglez_pl_df.group_by_dynamic(
            index_column="time", every="5m"
        ).agg([pl.all().exclude(["time"])])
        flag = anglez_grouped_by_window_length["angz_diff"].map_elements(
            lambda grouped_list_data: all(
                list_data < threshold_degrees for list_data in grouped_list_data
            )
        )

        return models.Measurement(
            measurements=(flag.to_numpy()).astype(int),
            time=anglez_grouped_by_window_length["time"],
        )

    def _find_onset_wakeup_times(
        self,
        spt_periods: List[Tuple[datetime.datetime, datetime.datetime]],
        sib_periods: List[Tuple[datetime.datetime, datetime.datetime]],
    ) -> List[SleepWindow]:
        """Find the sleep onset and wake up times.

        This function is implemented as follows:
        First, we use the spt_window as a guider for potential sleep.
        Then we find overlapping sustained inactivity bouts (sib_periods == 1).
        Sleep onset is defined as the start of the first sib_period that overlaps
        with a specific spt_window.
        Sleep wakeup is the end of the last SIB that overlaps with a spt window.

        Args:
            spt_periods: the sleep period guider windows, this is computed from the
                _find_periods method, it is a list of tuples of start and end times
                of the periods.
            sib_periods: the sustained inactivity bouts,  this is computed from the
                _find_periods method, it is a list of tuples of start and end times
                of the periods.

        Returns:
            A SleepWindow instance with sleep onset and wake up times.
            If there is no overlap between spt_windows and sib_periods,
            the onset and wakeup lists will be empty.
        """
        logger.debug(
            "Finding SIB periods within SPT windows. SPT periods:%s, SIB periods: %s",
            spt_periods,
            sib_periods,
        )
        sleep_windows = []
        for sleep_guide in spt_periods:
            min_onset = None
            max_wakeup = None
            for inactivity_bout in sib_periods:
                if (
                    inactivity_bout[0] <= sleep_guide[1]
                    and inactivity_bout[1] >= sleep_guide[0]
                ):
                    if min_onset is None or inactivity_bout[0] < min_onset:
                        min_onset = inactivity_bout[0]
                    if max_wakeup is None or inactivity_bout[1] > max_wakeup:
                        max_wakeup = inactivity_bout[1]
            if min_onset is not None and max_wakeup is not None:
                sleep_windows.append(SleepWindow(onset=min_onset, wakeup=max_wakeup))
        logger.debug("Sleep windows found: %s", len(sleep_windows))
        return sleep_windows

    def _fill_false_blocks(
        self, boolean_array: np.ndarray, gap_block: int
    ) -> np.ndarray:
        """Helper function to fill gaps in SPT window that are less than gap_blocks.

        We find the first non-zero in the sleep_idx_array, if there are none ,
        we return the initial array.
        We then iterate over the array and count every zero between ones
        (skipping the leading zeros),
        if that value is less than the gap_block, we fill in with ones.

        Args:
            boolean_array: A generic boolean array, typically the SPT window.
            gap_block: the length of the gap that needs to be filled.

        Returns:
            A numpy array with 1s, typically for identified SPT windows.
        """
        n_zeros = 0
        first_one_idx = next(
            (index for index, value in enumerate(boolean_array) if value), None
        )
        if first_one_idx is None:
            return boolean_array

        for sleep_array_idx in range(first_one_idx, len(boolean_array)):
            sleep_value = boolean_array[sleep_array_idx]
            if not sleep_value:
                n_zeros += 1
                continue

            if n_zeros < gap_block:
                boolean_array[sleep_array_idx - n_zeros : sleep_array_idx] = True
            n_zeros = 0

        return boolean_array

    def _compute_abs_diff_mean_anglez(
        self, anglez_data: models.Measurement, window_size_seconds: int = 5
    ) -> models.Measurement:
        """Helper function to compute the absolute difference of averaged anglez data.

        Args:
            anglez_data: the raw anglez data.
            window_size_seconds: the window size in seconds to average the anglez data.

        Returns:
            A Measurement instance with the absolute difference of the anglez data.
            Note that the length of the returned Measurement instance will be one
            less than the input anglez_data, this is because np.diff returns an array
            that is diff_size shorter than the input, where diff_size is the size of
            the difference step.
        """
        anglez_epoch1 = computations.moving_mean(anglez_data, window_size_seconds)
        absolute_diff = np.abs(np.diff(anglez_epoch1.measurements))

        return models.Measurement(
            measurements=absolute_diff, time=anglez_epoch1.time[1:]
        )


def _find_periods(
    window_measurement: models.Measurement,
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """Find periods where window_measurement is equal to 1.

    This is a helper function to return the periods in the format of
    List [start_of_period, end_of_period], it is used in the
    GGIRSleepDetection class and when removing non-wear periods from sleep windows.

    Args:
        window_measurement: the Measurement instance, intended to be
            either the spt_window or sib_period.

    Returns:
        A list of tuples, where each tuple contains the start and end times of
        a period. For isolated ones the function returns the same start
        and end time. The list is sorted by time.
    """
    logger.debug("Finding periods in window measurement.")
    edge_detection = np.convolve([1, 3, 1], window_measurement.measurements, "same")
    single_one = np.nonzero(edge_detection == 3)[0]

    single_periods = [
        (window_measurement.time.item(idx), window_measurement.time.item(idx))
        for idx in single_one
    ]

    blocked_one_edge = np.nonzero(edge_detection == 4)[0]
    block_pairs = np.reshape(blocked_one_edge, (-1, 2))
    block_periods = [
        (window_measurement.time.item(idx[0]), window_measurement.time.item(idx[1]))
        for idx in block_pairs
    ]
    all_periods = single_periods + block_periods
    all_periods.sort()

    logger.debug("Found %s periods.", len(all_periods))
    return all_periods


def remove_nonwear_from_sleep(
    non_wear_array: models.Measurement,
    sleep_windows: List[SleepWindow],
) -> List[SleepWindow]:
    """Remove non-wear periods from sleep windows.

    This function finds all the non-wear periods and removes any sleep windows that have
    any overlap with the non-wear periods (including non-wear completely
    within the sleep window).

    Args:
        non_wear_array: The non-wear array generated from metrics.detect_nonwear
        sleep_windows : The list of sleep windows, where the entries are
            instances of the SleepWindow class.

    Returns:
        A List of the filtered sleep windows.
    """
    logger.debug(
        "Finding non-wear periods that overlap with any of the %s sleep windows.",
        len(sleep_windows),
    )
    nonwear_periods = _find_periods(non_wear_array)

    filtered_sleep_windows = []
    for sleep_window in sleep_windows:
        if isinstance(sleep_window.onset, datetime.datetime) and isinstance(
            sleep_window.wakeup, datetime.datetime
        ):
            if not any(
                (
                    nonwear_period[0] <= sleep_window.onset <= nonwear_period[1]
                    or nonwear_period[0] <= sleep_window.wakeup <= nonwear_period[1]
                )
                or (
                    sleep_window.onset <= nonwear_period[0]
                    and sleep_window.wakeup >= nonwear_period[1]
                )
                for nonwear_period in nonwear_periods
            ):
                filtered_sleep_windows.append(sleep_window)

    logger.debug("Non-wear removed. %s sleep windows remain.", len(sleep_windows))
    return filtered_sleep_windows


def compute_physical_activty_categories(
    enmo_epoch1: models.Measurement,
    thresholds: Tuple[float, float, float] = (0.0563, 0.1916, 0.6958),
) -> models.Measurement:
    """Compute the physical activity categories based on the ENMO data.

    This function uses the enmo_epoch1 data (5s aggregated data) to compute three
    physical activity levels: light, moderate, and vigorous.

    Default values are taken from the Hildebrand 2014 study, and is best suited for
    children aged 7 - 11 years.

    Args:
        enmo_epoch1: The enmo epoch1 data, as physical activity data should be computed
            on aggregated data.
        thresholds: The threshold values for the physical activity categories.
            The default values are
                (light_threshold, moderate_threshold, vigorous_threshold).

    Returns:
        A Measurement instance with the physical activity categories. Categories are
        Inactive, Light, Moderate, Vigorous. The temporal resolution is the same as
        enmo_epoch1.

    Raises:
        ValueError: If the threshold values are not poisitive, unique and  in ascending
        order.

    References:
        Hildebrand, M., et al. Age group comparability of raw accelerometer output
            from wrist- and hip-worn monitors. Medicine and Science in
            Sports and Exercise, 46(9), 1816-1824 (2014).
            https://doi.org/10.1249/mss.0000000000000289
    """
    logger.debug("Computing physical activity levels, thresholds: %s", thresholds)
    if not (0 <= thresholds[0] < thresholds[1] < thresholds[2]):
        message = "Thresholds must be positive, unique, and given in ascending order."
        logger.error(message)
        raise ValueError(message)

    activity_levels = np.select(
        [
            enmo_epoch1.measurements <= thresholds[0],
            (thresholds[0] < enmo_epoch1.measurements)
            & (enmo_epoch1.measurements <= thresholds[1]),
            (thresholds[1] < enmo_epoch1.measurements)
            & (enmo_epoch1.measurements <= thresholds[2]),
            thresholds[2] < enmo_epoch1.measurements,
        ],
        ["inactive", "light", "moderate", "vigorous"],
    )
    return models.Measurement(measurements=activity_levels, time=enmo_epoch1.time)
