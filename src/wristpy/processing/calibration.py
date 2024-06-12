"""Calibrate accelerometer data."""

from typing import Optional

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


class InsufficientHours(Exception):
    """Raises error when not enough data hours."""

    def __init__(self, min_calibration_hours: int, total_hours: int) -> None:
        """Init."""
        self.message = (
            f"Calibration requires {min_calibration_hours} hours, "
            f"only {total_hours} hours available."
        )
        super().__init__(self.message)


class Calibration:
    """Calibration."""

    def __init__(
        self,
        chunked: bool = False,
        min_acceleration: float = 0.3,
        min_calibration_hours: int = 72,
        min_standard_deviation: float = 0.013,
        max_iterations: int = 1000,
        error_tolerance: float = 1e-10,
        min_error: float = 0.01,
        sampling_rate_hz: int = 30,
    ) -> None:
        """init."""
        self.chunked = chunked
        self.min_acceleration = min_acceleration
        self.min_calibration_hours = min_calibration_hours
        self.min_standard_deviation = min_standard_deviation
        self.max_iterations = max_iterations
        self.error_tolerance = error_tolerance
        self.min_error = min_error
        self.sampling_rate_hz: int = sampling_rate_hz
        self.current_iteration = 0

    def _calibrate(self, acceleration: models.Measurement) -> models.Measurement:
        """calibrates."""
        moving_sd = computations.moving_std(acceleration, 10)
        moving_mean = computations.moving_mean(acceleration, 10)

        no_motion_check = np.all(
            moving_sd.measurements < self.min_standard_deviation, axis=1
        ) & np.all(np.abs(moving_mean.measurements) < 2, axis=1)

        offset = np.zeros(3)
        scale = np.ones(3)

        no_motion_data = moving_mean.measurements[no_motion_check]

        sphere_criteria_check = np.sum(
            (no_motion_data.min(axis=0) < -self.min_acceleration)
            & (no_motion_data.max(axis=0) > self.min_acceleration)
        )
        if sphere_criteria_check != 3:
            raise SphereCriteriaError

        weights = np.ones(no_motion_data.shape[0]) * 100
        residual = [np.Inf]

        linear_regression_model = linear_model.LinearRegression()

        cal_error_initial = np.round(
            np.mean(abs(np.linalg.norm(no_motion_data, axis=1) - 1)), decimals=5
        )

        for i in range(self.max_iterations):
            current = (no_motion_data * scale) + offset
            closest_point = current / np.linalg.norm(current, axis=1, keepdims=True)
            offset_change = np.zeros(3)
            scale_change = np.ones(3)

            for k in range(3):
                x_ = np.vstack((current[:, k]))
                tmp_y = np.vstack((closest_point[:, k]))
                linear_regression_model.fit(x_, tmp_y, sample_weight=weights)

                offset_change[k] = linear_regression_model.intercept_[0]
                scale_change[k] = linear_regression_model.coef_[0, 0]
                current[:, k] = (x_ @ linear_regression_model.coef_).flatten()

            scale = scale_change * scale
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

        no_motion_calibrated = (no_motion_data * scale) + offset
        cal_error_end = np.around(
            np.mean(abs(np.linalg.norm(no_motion_calibrated, axis=1) - 1)), decimals=5
        )

        if (cal_error_end >= cal_error_initial) or (cal_error_end >= self.min_error):
            raise CalibrationError

        return {"scale": scale, "offset": offset}

    def _chunk(self, acceleration: models.Measurement) -> Optional[models.Measurement]:
        """Chunk data until you have enough to calibrate."""
        min_samples = int(self.min_calibration_hours * 3600 * self.sampling_rate_hz)
        chunk_size = int(12 * 3600 * self.sampling_rate_hz)

        chunk_num = 0

        total_samples = acceleration.measurements.shape[0]
        current_samples = min_samples

        while current_samples <= total_samples:
            accel_data_trimmed = acceleration.measurements[
                : min_samples + (chunk_size * chunk_num), :
            ]
            time_data_trimmed = acceleration.time[
                : min_samples + (chunk_size * chunk_num)
            ]
            try:
                return self._calibrate(
                    models.Measurement(
                        measurements=accel_data_trimmed, time=time_data_trimmed
                    )
                )
            except (SphereCriteriaError, CalibrationError):
                chunk_num += 1
                current_samples += chunk_size

        raise CalibrationError

    def run(self, acceleration: models.Measurement) -> Optional[models.Measurement]:
        """Based on the settings run."""
        data_range = acceleration.time.max() - acceleration.time.min()
        total_hours = data_range.total_seconds() / 3600

        if total_hours < self.min_calibration_hours:
            raise InsufficientHours(self.min_calibration_hours, total_hours)

        try:
            if self.chunked:
                linear_trans = self._chunk(acceleration)
            else:
                linear_trans = self._calibrate(acceleration=acceleration)
        except (SphereCriteriaError, CalibrationError) as e:
            print(e)
            return None

        transformed_data = (
            acceleration.measurements * linear_trans["scale"]
        ) + linear_trans["offset"]

        return models.Measurement(measurements=transformed_data, time=acceleration.time)
