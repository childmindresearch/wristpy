"""Calibrate accelerometer data."""

import abc
import math
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Union, cast

import numpy as np
import polars as pl
from scipy import optimize
from sklearn import linear_model
from sklearn import metrics as sklearn_metrics

from wristpy.core import computations, config, exceptions, models

logger = config.get_logger()


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


class AbstractCalibrator(abc.ABC):
    """Abstract class defining the interface for the different calibration methods."""

    @abc.abstractmethod
    def __init__(self) -> None:
        """Initialization function for the calibrator."""
        pass

    @abc.abstractmethod
    def run_calibration(self, acceleration: models.Measurement) -> models.Measurement:
        """The calibration method must contain a run_calibration function.

        The function must take the acceleration measurement object as input and return
        a measurement object that contains the calibrated acceleration data.
        """
        pass


class GgirCalibration(AbstractCalibrator):
    """Implements the GGIR calibration on accelerometer data.

    This class implements methods for autocalibrating accelerometer data using either
    entire dataset or subsets, as determined by the settings. Depending on the settings
    a scale and offset value is determined and applied to the data which, if
    successful, is returned to the user.


    Attributes:
        chunked: If true will perform calibration on subsets of data if false will
            calibrate on entire dataset.
        min_acceleration: Minimum acceleration for sphere criteria, in g-force.
        min_calibration_hours: Minimum hours of data required for calibration.
        no_motion_threshold: Minimum standard deviation for no-motion detection.
        max_iterations: Maximum number of iterations for optimization.
        error_tolerance: Tolerance for optimization convergence.
        max_calibration_error: Threshold for the maximum allowable calibration error.

    """

    def __init__(
        self,
        chunked: bool = False,
        min_acceleration: float = 0.3,
        min_calibration_hours: int = 72,
        no_motion_threshold: float = 0.013,
        max_iterations: int = 1000,
        error_tolerance: float = 1e-10,
        max_calibration_error: float = 0.01,
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
                perform calibration. Default is 72. If chunked calibration is selected
                this will be the size of the initial subset taken for calibration. If
                error has not been reduced below the max_calibration_error value, an
                additional 12 hours will be taken until all data is used or calibration
                is successful.
            no_motion_threshold: The standard deviation criteria used to select
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
            max_calibration_error: Maximum acceptable calibration error. If calibration
                cannot reach this threshold it will error.

        Returns:
            None
        """
        self.chunked = chunked
        self.min_acceleration = min_acceleration
        self.min_calibration_hours = min_calibration_hours
        self.no_motion_threshold = no_motion_threshold
        self.max_iterations = max_iterations
        self.error_tolerance = error_tolerance
        self.max_calibration_error = max_calibration_error

    def run_calibration(self, acceleration: models.Measurement) -> models.Measurement:
        """Runs calibration on acceleration data based on settings.

        If the chunked argument is set to true, it will run calibration on an initial
        chunk of the data the size of which is set to min_calibration_hours. If it fails
        to calibrate based on this subset, it will add 12-hour chunks to the subset
        until calibration succeeds, or fails on the entire dataset. Otherwise, if
        chunked is false (default) calibration will be conducted on the whole of the
        dataset. Calibration is successful when scale and offset values, that
        sufficiently minimize error when applied to the data, are found. The scale and
        offset values are then applied to every data point in the dataset which is
        then returned as calibrated models.Measurement object.

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
                `max_calibration_error` threshold.
        """
        logger.debug("Starting calibration.")
        data_range = cast(datetime, acceleration.time.max()) - cast(
            datetime, acceleration.time.min()
        )
        total_hours = math.floor(data_range.total_seconds() / 3600)

        if total_hours < self.min_calibration_hours:
            raise exceptions.CalibrationError(
                (
                    f"Calibration requires {self.min_calibration_hours} hours"
                    f"but only {total_hours} hours of data were given."
                ),
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
                process fails to get below the `max_calibration_error` threshold.
        """
        logger.debug("Running chunked calibration.")
        for chunk in self._get_chunk(acceleration):
            try:
                return self._calibrate(chunk)
            except (
                exceptions.SphereCriteriaError,
                exceptions.CalibrationError,
                exceptions.NoMotionError,
            ):
                pass
        raise exceptions.CalibrationError(
            "After all chunks of data used calibration has failed."
        )

    def _get_chunk(
        self, acceleration: models.Measurement
    ) -> Generator[models.Measurement, None, None]:
        """Takes a subset of acceleration data to be used for calibration.

        Args:
            acceleration: the accelerometer data containing x,y,z axis
                data and time stamps.

        Returns:
            The minimum hours of accelerometer data + additional 12 hour chunks of data
            every time the generator function is called.

        """
        logger.debug("Getting chunk.")
        sampling_rate = GgirCalibration._get_sampling_rate(timestamps=acceleration.time)
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
                error exceeds the `max_calibration_error` threshold.

        References:
            van Hees VT, Fang Z, Langford J, et al. Autocalibration of accelerometer
            data for free-living physical activity assessment using local gravity
            and temperature: an evaluation on four continents. J Appl Physiol (1985)
            2014 Oct 1;117(7):738-44. doi: 10.1152/japplphysiol.00421.2014.
        """
        logger.debug("Attempting to calibrate...")
        no_motion_data = _extract_no_motion(acceleration=acceleration)

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
            cal_error_end >= self.max_calibration_error
        ):
            raise exceptions.CalibrationError(
                "Calibration error could not be sufficiently minimized. "
                f"Initial Error: {cal_error_initial}, Final Error: {cal_error_end}, "
                f"Error threshold: {self.max_calibration_error}"
            )
        logger.debug(
            "Calibration successful. Scale: %s, Offset: %s",
            linear_transformation.scale,
            linear_transformation.offset,
        )
        return linear_transformation

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
        logger.debug("Beginning closest point fit.")
        sphere_criteria_check = np.all(
            (no_motion_data.min(axis=0) < -self.min_acceleration)
            & (no_motion_data.max(axis=0) > self.min_acceleration)
        )
        if not sphere_criteria_check:
            raise exceptions.SphereCriteriaError(
                "Did not meet criteria to sufficiently populate sphere"
            )

        weights = np.ones(no_motion_data.shape[0]) * 100
        previous_residual = np.inf

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
                raise exceptions.CalibrationError(
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

            logger.debug("Scale: %s, Offset: %s, Residual: %s", scale, offset, residual)
            if abs(residual - previous_residual) < self.error_tolerance:
                logger.debug("Change in residual below error tolerance, ending loop.}")
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
            (
                cast(datetime, timestamps.max()) - cast(datetime, timestamps.min())
            ).total_seconds()
        )

        return round(sampling_rate)


class ConstrainedMinimizationCalibration(AbstractCalibrator):
    """Calibrates accelerometer data using the default wristpy method.

    This is a modification of the method proposed by Van Hees et al. (2014), in which we
    use all available data, make use of the scipy optimize minimize function to solve
    for the scaling and offset parameters of a LinearTransformation that will be applied
    to the data. The minimization function looks to minimize the error between the
    no_motion_data and the unit sphere.

    Attributes:
        acceleration: The accelerometer data that we want to calibrate.
        no_motion_threshold: Minimum standard deviation for no-motion detection.
    """

    def __init__(
        self,
        no_motion_threshold: float = 0.013,
        max_iterations: int = 1000,
        max_calibration_error: float = 0.01,
    ) -> None:
        """Initializes class.

        Args:
            no_motion_threshold: The standard deviation criteria used to find
                periods of no motion. Default is 0.013g.
            max_iterations: The maximum amount of iterations for the
                closest_point_fit function.
            max_calibration_error: Maximum acceptable calibration error. If calibration
                cannot reach this threshold it will error.

        Returns:
            None
        """
        self.no_motion_threshold = no_motion_threshold
        self.max_iterations = max_iterations
        self.max_calibration_error = max_calibration_error

    def run_calibration(self, acceleration: models.Measurement) -> models.Measurement:
        """Runs calibration on acceleration data.

        Args:
            acceleration: the accelerometer data containing x,y,z axis
                data and time stamps.

        Returns:
            A Measurement object that contains the calibrated acceleration data.

        Raises:
            CalibrationError: If the calibration process fails to converge or the final
                error exceeds the `max_calibration_error` threshold.
        """
        logger.debug("Starting calibration.")
        no_motion_data = _extract_no_motion(
            acceleration=acceleration,
            no_motion_threshold=self.no_motion_threshold,
        )
        linear_transformation = self._closest_point_fit(no_motion_data=no_motion_data)

        logger.debug(
            "Calibration successful. Scale: %s, Offset: %s. ",
            linear_transformation.scale,
            linear_transformation.offset,
        )

        calibrated_acceleration = (
            acceleration.measurements * linear_transformation.scale
        ) + linear_transformation.offset

        return models.Measurement(
            measurements=calibrated_acceleration, time=acceleration.time
        )

    def _closest_point_fit(self, no_motion_data: np.ndarray) -> LinearTransformation:
        """Find linear transformation parameters that minimizes distance to unit sphere.

        This function implements the scipy optimize.minimize function to find the
        optimal scale and offset parameters that will minimize the error function,
        which is defined as the distance of the no_motion acceleration data
        from the unit sphere (where the unit sphere represents the ideal data points
        under 1g acceleration due to Earth's gravity). The initial guess for the
        scale and offset are chosen as 1/0, we provide constrained bounds for the
        scale and offset parameters to avoid the case where the scale can be
        set to 0 and offset to 1/sqrt(3) (exact unit sphere).

        Args:
            no_motion_data: The acceleration data during periods of no motion.

        Returns:
            A LinearTransformation object with scale and offset parameters to be applied
            to acceleration data for calibration.

        Raises:
            CalibrationError: If the optimization process fails to converge or if the
                calibration error is not sufficiently minimized.
        """
        start_scale = np.ones(3)
        start_offset = np.zeros(3)

        def get_distance_to_unit_sphere(params: list[float]) -> float:
            scale = params[:3]
            offset = params[3:]
            distances = np.linalg.norm((no_motion_data * scale) + offset, axis=1) - 1
            return np.sum(distances**2)

        initial_guess = np.concatenate([start_scale, start_offset])
        data_range = no_motion_data.max() - no_motion_data.min()
        scale_bounds = [(0.1, None)] * 3
        offset_bounds = [(-0.5 * data_range, 0.5 * data_range)] * 3
        bounds = scale_bounds + offset_bounds
        result = optimize.minimize(
            get_distance_to_unit_sphere,
            initial_guess,
            options={"maxiter": self.max_iterations},
            bounds=bounds,
        )

        if result.success:
            optimal_scale = result.x[:3]
            optimal_offset = result.x[3:]
            cal_error_end = np.sqrt(result.fun / len(no_motion_data))
        else:
            raise exceptions.CalibrationError("Optimization failed.")

        if cal_error_end >= self.max_calibration_error:
            cal_error_initial = np.mean(
                (np.linalg.norm(no_motion_data, axis=1) - 1) ** 2
            )
            logger.debug(
                "Calibration error could not be sufficiently minimized."
                "Initial Error: %s,  Final Error: %s, Error threshold: %s.",
                cal_error_initial,
                cal_error_end,
                self.max_calibration_error,
            )
            raise exceptions.CalibrationError(
                "Calibration error could not be sufficiently minimized."
            )

        return LinearTransformation(scale=optimal_scale, offset=optimal_offset)


def _extract_no_motion(
    acceleration: models.Measurement, no_motion_threshold: float = 0.013
) -> np.ndarray:
    """Identifies areas of stillness using standard deviation and mean.

    The function first takes a moving standard deviation and a moving mean of the
    acceleration data in 10 second epochs. These ndarrays are then used to identify
    the portions of the data that have a standard deviation below
    no_motion_threshold and a mean value < 2. These epochs are determined to be
    the periods where the accelerometer was influenced by no motion beyond the
    force of gravity. If periods of no motion are found, the ndarray is returned,
    to be fit to the idealized unit sphere for the purposes of calibration.

    Args:
        acceleration: the accelerometer data containing x,y,z axis
            data and time stamps.
        no_motion_threshold: Threshold for the standard deviaton of acceleration data
            used to find periods of no motion.

    Returns:
        An ndarray containing the accelerometer data determined to have no motion.

    Raises:
        NoMotionError: If no portions of data meet no motion criteria as defined by
            no_motion_check.

    References:
        van Hees VT, Fang Z, Langford J, et al. Autocalibration of accelerometer
        data for free-living physical activity assessment using local gravity
        and temperature: an evaluation on four continents. J Appl Physiol (1985)
        2014 Oct 1;117(7):738-44. doi: 10.1152/japplphysiol.00421.2014.
    """
    logger.debug("Extracting no motion.")
    moving_sd = computations.moving_std(acceleration, 10)
    moving_mean = computations.moving_mean(acceleration, 10)
    no_motion_check = np.all(
        moving_sd.measurements < no_motion_threshold, axis=1
    ) & np.all(np.abs(moving_mean.measurements) < 2, axis=1)

    no_motion_data = moving_mean.measurements[no_motion_check]

    if not np.any(no_motion_data):
        raise exceptions.NoMotionError(
            "Zero non-motion epochs found. Data did not meet criteria."
        )

    return no_motion_data


class CalibrationDispatcher:
    """Class used to select and implement appropriate calibrator."""

    _calibrator: Union[GgirCalibration, ConstrainedMinimizationCalibration]

    def __init__(self, name: Literal["ggir", "gradient"]) -> None:
        """Initializes the calibrator to one of the predefined calibrators.

        Args:
            name: The name of the calibrator to use. Options are "ggir" or "gradient".

        Raises:
            ValueError: If an unknown calibrator is given
        """
        if name == "ggir":
            self._calibrator = GgirCalibration()
        elif name == "gradient":
            self._calibrator = ConstrainedMinimizationCalibration()
        else:
            raise ValueError("Unknown calibrator.")

    def run(
        self, acceleration: models.Measurement, *, return_input_on_error: bool = False
    ) -> models.Measurement:
        """Runs calibration on acceleration data.

        Args:
            acceleration: the accelerometer data containing x,y,z axis
                data and time stamps.
            return_input_on_error: If true, will return the input acceleration data
                if calibration fails. If false, will raise an exception.

        Returns:
            A Measurement object that contains the calibrated acceleration data.

        Raises:
            CalibrationError: If the calibration process fails to converge or the final
                error exceeds the `max_calibration_error` threshold.
            SphereCriteriaError: If the sphere is not sufficiently populated, i.e. every
                axis does not have at least 1 value both above and below  the + and
                - value of min_acceleraiton.
            NoMotionError: If no portions of data meet no motion criteria as defined by
                no_motion_check.
        """
        try:
            return self._calibrator.run_calibration(acceleration)
        except (
            exceptions.CalibrationError,
            exceptions.SphereCriteriaError,
            exceptions.NoMotionError,
        ) as exc_info:
            if not return_input_on_error:
                raise
            logger.error(
                "Calibration FAILED: %s. Proceeding without calibration.", exc_info
            )
            return acceleration
