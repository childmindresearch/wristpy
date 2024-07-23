"""Calibrate accelerometer data."""

import math
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn import linear_model
from sklearn import metrics as sklearn_metrics

from wristpy.core import computations, models


class SphereCriteriaError(Exception):
    """Data did not meet the sphere criteria."""

    pass


class CalibrationError(Exception):
    """Was not able to lower calibration below error threshold."""

    pass


class NoMotionError(Exception):
    """No epochs with zero movement could be found in the data."""

    pass


class ZeroScaleError(Exception):
    """Scale value went to zero."""

    pass


@dataclass
class LinearTransformation:
    """Data class that contains scale and offset values for calibration.

    Attributes:
        scale: 3-element 1-D array of floats used to scale accelerameter data for
            calibration. Each element corresponds to the scaling factor for each axis.
        offset: 3-element 1-D array of floats used to offset accelerameter data for
            calibration. Each element corresponds to the offset value for each axis.
    """

    scale: np.ndarray
    offset: np.ndarray


class Calibration:
    """Implements calibration on accelerometer data, based off of GGIR's implementation.

    This class implements methods for autocalibrating accelerometer data using either
    entire dataset or subsets, as determined by the settings. Depending on the settings
    a scale and offset value is determined and applied to the data which, if
    successful, is returned to the user.


    Attributes:
            chunked: If true will preform calibration on subsets of data if false will
                calibrate on entire dataset.
            min_acceleration: Minimum acceleration for sphere criteria.
            min_calibration_hours: Minimum hours of data required for calibration.
            min_standard_deviation: Minimum standard deviation for no-motion detection.
            max_iterations: Maximum number of iterations for optimization.
            error_tolerance: Tolerance for optimization convergence.
            min_calibration_error: Threshold for calibration error.

    """

    def __init__(
        self,
        chunked: bool = False,
        min_acceleration: float = 0.3,
        min_calibration_hours: int = 72,
        min_standard_deviation: float = 0.013,
        max_iterations: int = 1000,
        error_tolerance: float = 1e-10,
        min_calibration_error: float = 0.01,
    ) -> None:
        """Initializes class.

        Attributes:
            chunked: Determines if either entire dataset will be used, or if
                calibration will be attempted with a subset of the data. Set to false by
                default, when true will initiate _chunked_calibration from the run()
                method instead of standard _calibration.
            min_acceleration: The value on either side of 0g for each axis.
                Determines if sphere is sufficiently populated to obtain meaningful
                calibration result. Default is 0.3g.
            min_calibration_hours: The minimum amount of data in hours needed to
                preform calibration. Default is 72. If chunked calibration is selected
                this will be the size of the initial subset taken for calibration. If
                error has not been reduced below the min_calibration_error value, an
                additional 12 hours will be taken until all data is used or calibration
                is successful.
            min_standard_deviation: The standard deviation critieria used to select
                portions of the data with no movement. Rolling windows with standard
                deviations below this value will be determined as "still". This value is
                likely to be different between devices. Default is 0.013g. This value
                was determined just above the empirically derived  baseline standard
                deviation due to noise (0.010g)(van Hees et al. 2014). If measuring the
                noise in a bench-top test, this threshold should be about `1.2 * noise`.
            max_iterations: The maximum amount of iterations for the closest_point_fit
                method. Default is 1000, generally should be left at this value.
            error_tolerance: Tolerated level of error, when the
                closest_point_fit method arrives at this value or better, the process
                ends. Default is 1e-10. Generally should be left at this value.
            min_calibration_error: Minimum acceptable error. If calibration can
                not reach this threshold it will error.

        Returns:
            None
        """
        self.chunked = chunked
        self.min_acceleration = min_acceleration
        self.min_calibration_hours = min_calibration_hours
        self.min_standard_deviation = min_standard_deviation
        self.max_iterations = max_iterations
        self.error_tolerance = error_tolerance
        self.min_calibration_error = min_calibration_error

    def run(self, acceleration: models.Measurement) -> models.Measurement:
        """Runs calibration on acceleration data based on settings.

        If the chunked arguement is set to true, it will run calibration on an initial
        chunk of the data the size of which is set to min_calibration_hours. If it fails
        to calibrate based on this subset, it will add 12 hour chunks to the subset
        until calibration succeeds, or fails on the entire dataset. Otherwise if chunked
        is false (default) calibration will be conducted on the whole of the dataset.
        Calibration is successful when scale and offset values, that sufficiently
        minimize error when applied to the data, are found. The scale and offset values
        are then applied to every data point in the dataset which is then returned as
        calibrated models.Measurement object.

        Args:
            acceleration: the accelerometer data containing x,y,z axis
                data and time stamps.

        Returns:
            A Measurement object that contains the calibrated acceleration data.

        Raises:
            ValueError: If the acceleration data does not meet the specified minimum
                hours of data as given by min_calibration_hours.
            SphereCriteriaError: If the sphere is not sufficiently populated, i.e. every
                axis does not have at least 1 value both above and below  the + and
                - value of min_acceleraiton.
            CalibrationError: If the calibration process fails to get below the
                `min_calibration_error` threshold.
        """
        data_range = acceleration.time.max() - acceleration.time.min()
        total_hours = math.floor(data_range.total_seconds() / 3600)

        if total_hours < self.min_calibration_hours:
            raise ValueError(
                f"Calibration requires {self.min_calibration_hours} hours",
                f"but only {total_hours} hours of data were given.",
            )

        if self.chunked:
            linear_transformation = self._chunked_calibration(acceleration=acceleration)
        else:
            linear_transformation = self._calibrate(acceleration=acceleration)

        calibrated_acceleration = (
            acceleration.measurements * linear_transformation.scale
        ) + linear_transformation.offset

        return models.Measurement(
            measurements=calibrated_acceleration, time=acceleration.time
        )

    def _chunked_calibration(
        self, acceleration: models.Measurement
    ) -> LinearTransformation:
        """Chunks the data into subsets, to calibrate on smaller sections of data.

        The first chunk is determined by the min_calibration_hours. Should calibration
        fail on that subset of data, additional chunks of data are added to attempt
        calibration on. This continues until all data is tried, or calibration succeeds.
        Chunks are added in increments of 12 hours.

        Args:
            acceleration: the accelerometer data containing x,y,z axis
                data and time stamps.

        Returns:
            A LinearTransformation object with scale and offset attributes to be applied
            to acceleration data for calibration.

        Raises:
            CalibrationError: If all possible chunks have been used and the calibration
                process fails to get below the `min_calibration_error` threshold.
        """
        for chunk in self._get_chunk(acceleration):
            try:
                return self._calibrate(chunk)
            except (SphereCriteriaError, CalibrationError, NoMotionError):
                pass
        raise CalibrationError("After all chunks of data used calibration has failed.")

    def _get_chunk(
        self, acceleration: models.Measurement
    ) -> Generator[models.Measurement, None, None]:
        """Takes a subset of acceleration data to be used for calibration.

        Args:
            acceleration: the accelerometer data containing x,y,z axis
                data and time stamps.

        Returns:
            The minimum hours of accelerometer data + additional 12 hour chunks of data
            everytime the generator function is called.

        """
        sampling_rate = Calibration._get_sampling_rate(timestamps=acceleration.time)
        min_samples = int(self.min_calibration_hours * 3600 * sampling_rate)
        chunk_size = int(12 * 3600 * sampling_rate)
        total_samples = len(acceleration.measurements)
        if min_samples == total_samples:
            sample_indices = [total_samples]
        else:
            sample_indices = list(range(min_samples, total_samples, chunk_size))
            if sample_indices[-1] != total_samples:
                sample_indices += [total_samples]

        for idx in sample_indices:
            yield models.Measurement(
                measurements=acceleration.measurements[:idx, :],
                time=acceleration.time[:idx],
            )

    def _calibrate(self, acceleration: models.Measurement) -> LinearTransformation:
        """Calibrates data and returns scale and offset values.

        The acceleration data is processed by the _extract_no_motion function, which
        returns the portions of the data where the subject was still. This data ideally
        should all have points with a norm of 1. If we take a unit sphere, all the
        points in no_motion_data should be found along it's surface. As this is not the
        case we calibrate by finding offset and scale values that will most closely make
        the whole of our data lie along this unit sphere. This is done by the
        _closest_point_fit function. We then transform the data by applying the scale
        and offset values we have found, and if the error has been sufficiently reduced,
        these values are returned. If not calibration fails and raises a calibration
        error.

        Args:
            acceleration: the accelerometer data containing x,y,z axis
                data and time stamps.

        Returns:
            A LinearTransformation object with scale and offset attributes to be applied
            to acceleration data for calibration.

        Raises:
            CalibrationError: If the calibration process fails to converge or the final
                error exceeds the `min_calibration_error` threshold.

        References:
            van Hees VT, Fang Z, Langford J, et al. Autocalibration of accelerometer
            data for free-living physical activity assessment using local gravity
            and temperature: an evaluation on four continents. J Appl Physiol (1985)
            2014 Oct 1;117(7):738-44. doi: 10.1152/japplphysiol.00421.2014.
        """
        no_motion_data = self._extract_no_motion(acceleration=acceleration)
        linear_transformation = self._closest_point_fit(no_motion_data=no_motion_data)

        no_motion_calibrated = (
            no_motion_data * linear_transformation.scale
        ) + linear_transformation.offset

        cal_error_initial = np.round(
            np.mean(abs(np.linalg.norm(no_motion_data, axis=1) - 1)), decimals=5
        )
        cal_error_end = np.around(
            np.mean(abs(np.linalg.norm(no_motion_calibrated, axis=1) - 1)), decimals=5
        )

        if (cal_error_end > cal_error_initial) or (
            cal_error_end >= self.min_calibration_error
        ):
            raise CalibrationError(
                "Calibration error could not be sufficiently minimized."
                f"Initial Error: {cal_error_initial}, Final Error: {cal_error_end},"
                f"Error threshold: {self.min_calibration_error}"
            )

        return linear_transformation

    def _extract_no_motion(self, acceleration: models.Measurement) -> np.ndarray:
        """Identifies areas of stillness using standard deviation and mean.

        The function first takes a moving standard deviation and a moving mean of the
        acceleration data in 10 second epochs. These ndarrays are then used to identify
        the portions of the data that have a standard deviation below
        min_standard_deviation and a mean value < 2. These epochs are determined to be
        the periods where the accelerometer was influenced by no motion beyond the
        force of gravity. If periods of no motion are found, the ndarray is returned,
        to be fit to the idealized unit sphere for the purposes of calibration

        Args:
            acceleration: the accelerometer data containing x,y,z axis
                data and time stamps.

        Returns:
            an ndarray containing the accelerometer data determined to have no motion.

        Raises:
            NoMotionError: If no portions of data meet no motion criteria as defined by
                no_motion_check.

        References:
            van Hees VT, Fang Z, Langford J, et al. Autocalibration of accelerometer
            data for free-living physical activity assessment using local gravity
            and temperature: an evaluation on four continents. J Appl Physiol (1985)
            2014 Oct 1;117(7):738-44. doi: 10.1152/japplphysiol.00421.2014.
        """
        moving_sd = computations.moving_std(acceleration, 10)
        moving_mean = computations.moving_mean(acceleration, 10)
        no_motion_check = np.all(
            moving_sd.measurements < self.min_standard_deviation, axis=1
        ) & np.all(np.abs(moving_mean.measurements) < 2, axis=1)

        no_motion_data = moving_mean.measurements[no_motion_check]

        if not np.any(no_motion_data):
            raise NoMotionError(
                "Zero non-motion epochs found. Data did not meet criteria."
            )

        return no_motion_data

    def _closest_point_fit(self, no_motion_data: np.ndarray) -> LinearTransformation:
        """Applies closest point fit to no motion data to calibrated accelerometer data.

        This method implements an iterative algorithm that finds the optimal scale and
        offset for calibrating accelerometer data. The ndarray it processes contains the
        periods of "no motion" identified by the _extract_no_motion method. This data is
        then fit to an idealized unit sphere which represents the accelerometer data
        under noiseless conditions. The scale and offset derived from this process are
        used to calibrate our data along each axis.

        Args:
            no_motion_data: The acceleration data during periods of no
                motion, in order to determine scale and offset.

        Returns:
            A LinearTransformation object with scale and offset attributes to be applied
            to acceleration data for calibration.

        Raises:
            SphereCriteriaError: If the sphere is not sufficiently populated, i.e. every
                axis does not have at least 1 value both above and below  the + and
                - value of min_acceleraiton.
            ZeroScaleError: Numerical instability inherent to the algorithm may cause
                values of the scale vector to tend toward zero under certain conditions.
                This will cause calibration to fail.

        References:
            van Hees VT, Fang Z, Langford J, et al. Autocalibration of accelerometer
            data for free-living physical activity assessment using local gravity
            and temperature: an evaluation on four continents. J Appl Physiol (1985)
            2014 Oct 1;117(7):738-44. doi: 10.1152/japplphysiol.00421.2014.
        """
        sphere_criteria_check = np.all(
            (no_motion_data.min(axis=0) < -self.min_acceleration)
            & (no_motion_data.max(axis=0) > self.min_acceleration)
        )
        if not sphere_criteria_check:
            raise SphereCriteriaError(
                "Did not meet criteria to sufficiently populate sphere"
            )

        weights = np.ones(no_motion_data.shape[0]) * 100
        previous_residual = np.Inf

        linear_regression_model = linear_model.LinearRegression()

        offset = np.zeros(3)
        scale = np.ones(3)
        for _ in range(self.max_iterations):
            current = (no_motion_data * scale) + offset
            closest_point = current / np.linalg.norm(current, axis=1, keepdims=True)
            linear_regression_model.fit(current, closest_point, sample_weight=weights)
            offset_change = linear_regression_model.intercept_
            scale_change = np.diag(linear_regression_model.coef_)

            if np.all((scale * scale_change) < 1e-8):
                raise ZeroScaleError(
                    """Calibration has failed as a result of zero scale values."""
                )

            scale *= scale_change
            offset += offset_change / scale

            residual = 0.03 * sklearn_metrics.mean_squared_error(
                current, closest_point, sample_weight=weights
            )

            weights = np.minimum(
                1 / np.linalg.norm(current - closest_point, axis=1), 100
            )

            if abs(residual - previous_residual) < self.error_tolerance:
                break

            previous_residual = residual

        return LinearTransformation(scale=scale, offset=offset)

    @staticmethod
    def _get_sampling_rate(timestamps: pl.Series) -> int:
        """Get the sampling rate.

        Args:
            timestamps: polars series of datetime objects representing the time points
            of each sample in the acceleration data.

        Returns:
            sampling rate in Hz.
        """
        sampling_rate = timestamps.len() / round(
            (timestamps.max() - timestamps.min()).total_seconds()  # type: ignore
        )

        return round(sampling_rate)
