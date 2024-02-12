"""Implement the GGIR calibration process."""

import numpy as np
import polars as pl
import sklearn.linear_model
from warnings import warn
from wristpy.common.data_model import InputData, OutputData
from wristpy.ggir.metrics_calc import moving_mean, moving_std


def start_ggir_calibration(
    input_data: InputData,
    sphere_crit: float = 0.3,
    min_hours: int = 72,
    sd_crit: float = 0.013,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> OutputData:
    """Applying the GGIR calibration procedure to raw accelerometer data.

    Args:
        input_data: the InputData class containing the raw data to calibrate

        sphere_crit : float, optional
            Minimum acceleration value (in g) on both sides of 0g for each axis. Determines if the
            sphere is sufficiently populated to obtain a meaningful calibration result.
            Default is 0.3g.

        min_hours : int, optional
            Ideal minimum hours of data to use for the calibration. Any values not factors of 12 are
            rounded up to the nearest factor. Default is 72. If less than this amout of data is
            avialable (but still more than 12 hours), calibration will still be performed on all the
            data. If the calibration error is not under 0.01g after these hours, more data will be used
            in 12 hour increments.

        sd_crit: float, optional
            The criteria for the rolling standard deviation to determine stillness, in g. This value
            will likely change between devices. Default is 0.013g, which was found for GeneActiv
            devices. If measuring the noise in a bench-top test, this threshold should be about
            `1.2 * noise`.

        max_iter : int, optional
            Maximum number of iterations to perform during calibration. Default is 1000. Generally
            should be left at this value.

        tol : float, optional
            Tolerance for stopping iteration. Default is 1e-10. Generally this should be left at this
            value.

    Returns:
        Output: added time column with corrected ms information and
        cast into datetime object
    """
    accel_data = input_data.acceleration
    time_data = input_data.time
    s_r = input_data.sampling_rate

    # parameters
    n10 = int(10 * s_r)  # samples in 10 seconds
    nh = int(min_hours * 3600 * s_r)  # samples in min_hours
    n12h = int(12 * 3600 * s_r)  # samples in 12 hours

    i_h = 0  # keep track of number of extra 12 hour blocks used

    # check if enough data
    if accel_data.height < nh:
        warn(
            f"Less than {min_hours} hours of data ({accel_data.height / (s_r * 3600)} hours). "
            f"No Calibration performed",
            UserWarning,
        )
        return {}

    acc_rsd = moving_std(accel_data)
    acc_rm = moving_mean(accel_data)
    # find periods of no motion
    no_motion = np.all(acc_rsd < sd_crit, axis=1) & np.all(abs(acc_rm) < 2, axis=1)

    # trim to no motion
    acc_rm_nm = acc_rm[no_motion]

    # flags for finished and if cal is valid
    finished = False
    valid_calibration = True

    # initialize offset and scale
    offset = np.zeros(3)
    scale = np.ones(3)

    if finished and valid_calibration:
        scaled_accel = apply_calibration(accel_data, scale, offset)

    return OutputData(cal_accel=scaled_accel, scale=scale, offset=offset)


def apply_calibration(accel_raw: InputData.acceleration, scale, offset) -> pl.DataFrame:
    """Apply calibration to raw data."""
    scaled_accel = (InputData.acceleration + offset) * scale

    return scaled_accel
