"""Calibrate accelerometer data."""

import math
import typing

import numpy as np
from sklearn import linear_model

from wristpy.core import computations, models


class SphereCriteriaError(Exception):
    """Raises error when calibration fails due to sphere criteria."""

    def __init__(self) -> None:
        """Initalize."""
        self.message = "Calibration failed, data did not meet sphere criteria."
        super().__init__(self.message)


class CalibrationError(Exception):
    """Raises error when calibration error can not get below error tolerance."""

    def __init__(self) -> None:
        """Init."""
        self.message = "Calibration failed, error could not be sufficiently improved"
        super().__init__(self.message)


class Calibration:
    """Implements calibration on accelerometer data, based off of GGIR's implementation.

    This class implements methods for autocalibrating accelerometer data using either
    entire dataset or subsets, as determined by the settings. Depending on the settings
    a scale and offset value is determined and applied to the data which, if
    successful, is returned to the user.

    Attributes.
        chunked (bool): Determines if either entire dataset will be used, or if
        calibration will be attempted with a subset of the data. Set to false by
        default, when true will initiate _chunked_calibration from the run() method
        instead of standard _calibration.

        min_acceleration (float): The value on either side of 0g for each axis.
        Determines if sphere is sufficiently populated to obtain meaningful calibration
        result. Default is 0.3g.

        min_calibration_hours (int): The minimum amount of data in hours needed to
        preform calibration. Default is 72. If chunked calibration is selected this will
        be the size of the initial subset taken for calibration.If error has not been
        reduced below the min_error value, an additional 12 hours will be taken until
        all data is used or calibration is successful.

        min_standard_deviation (float): The standard deviation critieria used to select
        portions of the data with no movement. Rolling windows with standard deviations
        below this value will be determined has "still". This value is likely to be
        different between devices. Default is 0.013g, the value determined for GeneActiv
        devices. If measuring the noise in a bench-top test, this threshold should be
        about `1.2 * noise`.

        max_iterations (int): The maximum amount of iterations for the closest_point_fit
        method. Default is 1000, generally should be left at this value.

        error_tolerance (float): Tolerated level of error, when the closest_point_fit
        method arrives at this value or better, the process ends. Default is 1e-10.
        Generally should be left at this value.

        min_error (float): Minimum acceptable error. If calibration can not reach this
        threshold it will error.




    """

    def __init__(
        self,
        chunked: bool = False,
        min_acceleration: float = 0.3,
        min_calibration_hours: int = 72,
        min_standard_deviation: float = 0.013,
        max_iterations: int = 1000,
        error_tolerance: float = 1e-10,
        min_error: float = 0.01,
    ) -> None:
        """init.

        Attributes:
            chunked (bool, optional): Whether to use chunked calibration for long
            recordings.

            min_acceleration (float,optional): Minimum acceleration for sphere criteria.

            min_calibration_hours (int,optional): Minimum hours of data required for
            calibration.

            min_standard_deviation (float,optional): Minimum standard deviation for
            no-motion detection.

            max_iterations (int,optional): Maximum number of iterations for
            optimization.

            error_tolerance (float,optional): Tolerance for optimization convergence.

            min_error (float,optional): Minimum acceptable calibration error.

        Returns:
            None
        """
        self.chunked = chunked
        self.min_acceleration = min_acceleration
        self.min_calibration_hours = min_calibration_hours
        self.min_standard_deviation = min_standard_deviation
        self.max_iterations = max_iterations
        self.error_tolerance = error_tolerance
        self.min_error = min_error
        self.current_iteration = 0

    def run(self, acceleration: models.Measurement) -> models.Measurement:
        """Runs acceleration data of the Measurement class based on the settings.

        Chunks the data if chunked = true, otherwise it preforms calibration on the
        entire dataset. linear transformation is then applied to the data.

        Args:
            acceleration(Measurement): the accelerometer data containing x,y,z axis
            data and time stamps.

        Returns:
            A Measurement object which the original input data, with a linear
            transformation applied.
        """
        data_range = acceleration.time.max() - acceleration.time.min()
        total_hours = math.ceil(data_range.total_seconds() / 3600)

        if total_hours < self.min_calibration_hours:
            raise ValueError(
                f"Calibration requires {self.min_calibration_hours} hours",
                f"but only {total_hours} hours of data were given.",
            )

        if self.chunked:
            linear_trans = self._chunked_calibration(acceleration=acceleration)
        else:
            linear_trans = self._calibrate(acceleration=acceleration)

        transformed_data = (
            acceleration.measurements * linear_trans["scale"]
        ) + linear_trans["offset"]

        return models.Measurement(measurements=transformed_data, time=acceleration.time)

    def _chunked_calibration(
        self, acceleration: models.Measurement
    ) -> dict[str, np.ndarray]:
        """Chunks the data into subsets, to calibrate on smaller sections of data.

        Args:
            acceleration (Measurement): the accelerometer data containing x,y,z axis
            data and time stamps.

        Returns:
            A dictionary with type str: ndarray. Contains two keys labeled `scale` and
            `offset`.
        """
        chunk_num = 0
        finished = False

        while not finished:
            subset = self._take_chunk(acceleration=acceleration, chunk_num=chunk_num)
            if len(acceleration.measurements) == len(subset.measurements):
                finished = True

            try:
                return self._calibrate(subset)
            except (SphereCriteriaError, CalibrationError, ValueError):
                chunk_num += 1

        raise CalibrationError

    def _take_chunk(
        self, acceleration: models.Measurement, chunk_num: int
    ) -> models.Measurement:
        """Takes the next chunk based on the chunk number.

        Args:
            acceleration (Measurement): the accelerometer data containing x,y,z axis
            data and time stamps.

            chunk_num (int): the current 12 hour chunk iteration being taken.

        Returns:
            The next chunk of the accelerometer data, if this is the final chunk, the
            entire data is returned.
        """
        sampling_rate = self._get_sampling_rate(acceleration=acceleration)
        min_samples = int(self.min_calibration_hours * 3600 * sampling_rate)
        chunk_size = int(12 * 3600 * sampling_rate)

        total_samples = len(acceleration.measurements)
        current_sample = min_samples + (chunk_num * chunk_size)

        if current_sample > total_samples:
            current_sample = total_samples

        return models.Measurement(
            measurements=acceleration.measurements[:current_sample, :],
            time=acceleration.time[:current_sample],
        )

    def _calibrate(self, acceleration: models.Measurement) -> dict[str, np.ndarray]:
        """Calibrates data and returns scale and offset values.

        If error is low enough, the linear transformation is returned in a dict,
        errors if not.

        Args:
            acceleration (Measurement): the accelerometer data containing x,y,z axis
            data and time stamps.

        Returns:
            A dictionary with type str: ndarray. Contains two keys labeled `scale` and
            `offset`.
        """
        no_motion_data = self._extract_no_motion(acceleration=acceleration)
        linear_tranformation = self._closest_point_fit(no_motion_data=no_motion_data)

        no_motion_calibrated = (
            no_motion_data * linear_tranformation["scale"]
        ) + linear_tranformation["offset"]

        cal_error_initial = np.round(
            np.mean(abs(np.linalg.norm(no_motion_data, axis=1) - 1)), decimals=5
        )
        cal_error_end = np.around(
            np.mean(abs(np.linalg.norm(no_motion_calibrated, axis=1) - 1)), decimals=5
        )

        if (cal_error_end >= cal_error_initial) or (cal_error_end >= self.min_error):
            raise CalibrationError

        return linear_tranformation

    def _extract_no_motion(self, acceleration: models.Measurement) -> np.ndarray:
        """Identifies areas of stillness using standard deviation and mean.

        Args:
            acceleration (Measurement): the accelerometer data containing x,y,z axis
            data and time stamps.

        Returns:
            an ndarray containing the accelerometer data determined to have no motion.
        """
        moving_sd = computations.moving_std(acceleration, 10)
        moving_mean = computations.moving_mean(acceleration, 10)

        no_motion_check = np.all(
            moving_sd.measurements < self.min_standard_deviation, axis=1
        ) & np.all(np.abs(moving_mean.measurements) < 2, axis=1)

        no_motion_data = moving_mean.measurements[no_motion_check]

        if no_motion_data.shape[0] == 0:
            raise ValueError(
                "Zero non-motion epochs found. Data did not meet criteria."
            )

        sphere_criteria_check = np.sum(
            (no_motion_data.min(axis=0) < -self.min_acceleration)
            & (no_motion_data.max(axis=0) > self.min_acceleration)
        )
        if sphere_criteria_check != 3:
            raise SphereCriteriaError
        else:
            return no_motion_data

    def _closest_point_fit(self, no_motion_data: np.ndarray) -> dict[str, np.ndarray]:
        """Applies closest point fit to no motion data.

        Args:
            no_motion_data (np.ndarray): The acceleration data during periods of no
            motion, in order to determine scale and offset.

        Returns:
            A dictionary with type str: ndarray. Contains two keys labeled `scale` and
            `offset`.
        """
        weights = np.ones(no_motion_data.shape[0]) * 100
        residual = [np.Inf]

        linear_regression_model = linear_model.LinearRegression()

        offset = np.zeros(3)
        scale = np.ones(3)
        for i in range(self.max_iterations):
            current = (no_motion_data * scale) + offset
            closest_point = current / np.linalg.norm(current, axis=1, keepdims=True)
            offset_change = np.zeros(3)
            scale_change = np.ones(3)

            for k in range(3):
                x_ = np.vstack(
                    typing.cast(
                        typing.Sequence[np.ndarray], (current[:, k].reshape(-1, 1))
                    )
                )
                tmp_y = np.vstack(
                    typing.cast(
                        typing.Sequence[np.ndarray],
                        (closest_point[:, k].reshape(-1, 1)),
                    )
                )
                linear_regression_model.fit(x_, tmp_y, sample_weight=weights)

                offset_change[k] = linear_regression_model.intercept_[0]
                scale_change[k] = linear_regression_model.coef_[0, 0]
                current[:, k] = (x_ @ linear_regression_model.coef_).flatten()

            scale = np.where(scale_change == 0, 1e-8, scale_change) * scale
            offset = offset_change + (offset / scale)

            residual.append(
                3
                * np.mean(
                    weights[:, None] * (current - closest_point) ** 2 / weights.sum()
                )
            )
            weights = np.minimum(
                1 / np.linalg.norm(current - closest_point, axis=1), 100
            )

            if abs(residual[i] - residual[i - 1]) < self.error_tolerance:
                break

        return {"scale": scale, "offset": offset}

    def _get_sampling_rate(self, acceleration: models.Measurement) -> int:
        """Get the sampling rate.

        Args:
            acceleration (Measurement): the accelerometer data containing x,y,z axis
            data and time stamps.

        Returns:
            sampling rate in Hz.
        """
        sampling_rate = acceleration.time.len() / round(
            (acceleration.time.max() - acceleration.time.min()).total_seconds()
        )
        return round(sampling_rate)
