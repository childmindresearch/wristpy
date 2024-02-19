"""Implement the GGIR calibration process."""

from warnings import warn

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
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
            Minimum acceleration value (in g) on both sides of 0g for each axis.
            Determines if thesphere is sufficiently populated to obtain a meaningful
            calibration result. Default is 0.3g.

        min_hours : int, optional
            Ideal minimum hours of data to use for the calibration. Any values not
            factors of 12 are rounded up to the nearest factor. Default is 72. If less
            than this amout of data is avialable (but still more than 12 hours),
            calibration will still be performed on all the data. If the calibration
            error is not under 0.01g after these hours, more data will be used in 12 hour increments.

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
        Output: Output data class with the calibrated acceleration data, the start and end calibration error,
        and the scale and offset value from calibration.
    """  # noqa: E501
    accel_data = input_data.acceleration
    time_data = input_data.time
    s_r = input_data.sampling_rate

    # parameters
    # n10 = int(10 * s_r)  # samples in 10 seconds, not used because hardcoded 10s
    nh = int(min_hours * 3600 * s_r)  # samples in min_hours
    n12h = int(12 * 3600 * s_r)  # samples in 12 hours

    i_h = 0  # keep track of number of extra 12 hour blocks used

    # check if enough data
    if accel_data.height < nh:
        warn(
            f"Less than {min_hours} hours of data ({accel_data.height / (s_r * 3600)} hours). "  # noqa: E501
            f"No Calibration performed",
            UserWarning,
        )
        return {}

    # flags for finished and if cal is valid
    finished = False
    valid_calibration = True

    while not finished:
        accel_data_trimmed = accel_data[: nh + i_h * n12h]
        (
            finished,
            offset,
            scale,
            cal_err_start,
            cal_err_end,
        ) = closest_point_fit(
            accel_data_trimmed, time_data, s_r, sd_crit, sphere_crit, max_iter, tol
        )

    if finished and valid_calibration:
        scaled_accel = apply_calibration(accel_data, scale, offset)

    return OutputData(
        cal_accel=scaled_accel,
        scale=scale,
        offset=offset,
        cal_error_start=cal_err_start,
        cal_error_end=cal_err_end,
    )


def apply_calibration(
    accel_raw: pl.DataFrame, scale: float, offset: float
) -> pl.DataFrame:
    """Apply calibration to raw data."""
    scaled_accel = accel_raw.select(
        [
            (pl.col(column_name) * scale[i] + offset[i]).alias(column_name)
            for i, column_name in enumerate(accel_raw.columns)
        ]
    )

    return scaled_accel


def closest_point_fit(
    accel_data: pl.DataFrame,
    time_data: pl.DataFrame,
    s_r: int,
    sd_crit: float,
    sphere_crit: float,
    max_iter: int,
    tol: float,
):
    """Do the iterative closest point fit.

    Args:
     accel_data:
     time_data:
     s_r:
     sd_crit:
     sphere_crit:
     max_iter:
     tol:

    Returns:
        Scale, offset, cal_error start, cal_error_end

    """
    # get the moving std and mean over a 10s window
    acc_rsd = moving_std(accel_data, time_data, s_r, 10)
    acc_rm = moving_mean(accel_data, time_data, s_r, 10)

    # grab only the accel data
    acc_rsd = acc_rsd.select(["X_std", "Y_std", "Z_std"])
    acc_rm = acc_rm.select(["X_mean", "Y_mean", "Z_mean"])
    # find periods of no motion
    no_motion = np.all(acc_rsd < sd_crit, axis=1) & np.all(abs(acc_rm) < 2, axis=1)

    # trim to no motion
    acc_rm_nm = acc_rm[no_motion]
    acc_rsd_nm = acc_rsd.filter(no_motion)

    # initialize offset and scale
    offset = np.zeros(3)
    scale = np.ones(3)

    tel = 0
    for col in acc_rm_nm.columns:
        tmp = (acc_rm_nm[col].min() < -sphere_crit) & (
            acc_rm_nm[col].max() > sphere_crit
        )
        if tmp:
            tel = tel + 1

    if tel != 3:
        return False, offset, scale

    offset = pl.Series(np.zeros(3))
    scale = pl.Series(np.ones(3).flatten())

    weights = np.ones(acc_rm_nm.shape[0]) * 100
    res = [np.Inf]
    LR = LinearRegression()

    acc_nm_pd = acc_rm_nm.to_pandas()
    curr = (acc_nm_pd * scale) + offset

    max_iter = 1000
    tol = 1e-10
    weights = np.ones(curr.shape[0]) * 100
    res = [np.Inf]

    for i in range(max_iter):
        curr = (acc_nm_pd * scale) + offset
        closest_point = curr / np.linalg.norm(curr, axis=1, keepdims=True)
        offsetch = np.zeros(3)
        scalech = np.ones(3)

        for k in range(3):
            # there was some code dropping NANs from closest point, but these should
            # be taken care of in the original mask. Division by zero should also
            # not be happening during motionless data, where 1 value should always be close
            # to 1
            x_ = np.vstack((curr.iloc[:, k]))
            tmp_y = np.vstack((closest_point.iloc[:, k]))
            LR.fit(x_, tmp_y, sample_weight=weights)

            offsetch[k] = LR.intercept_
            scalech[k] = LR.coef_[0]
            curr.iloc[:, k] = x_ @ LR.coef_

        scale = scalech * scale
        offset = offsetch + (offset / scale)
        res.append(
            3 * np.mean(weights[:, None] * (curr - closest_point) ** 2 / weights.sum())
        )
        weights = np.minimum(1 / np.linalg.norm(curr - closest_point, axis=1), 100)

        if abs(res[i] - res[i - 1]) < tol:
            break

    acc_cal_pd = (acc_nm_pd * scale) + offset
    cal_error_end = np.around(
        np.mean(abs(np.linalg.norm(acc_cal_pd, axis=1) - 1)), decimals=5
    )