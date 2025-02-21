"""test the calibration module."""

from datetime import datetime, timedelta
from typing import Union

import numpy as np
import polars as pl
import pytest

from wristpy.core import exceptions, models
from wristpy.processing import calibration


def create_dummy_measurement(
    sampling_rate: int,
    duration_hours: float = 84.0,
    all_same_num: Union[float, None] = None,
) -> models.Measurement:
    """Create dummy measurement for testing.

    Args:
        sampling_rate: How many samples per second are to be created.
        duration_hours: How many hours of data are to be created.
        all_same_num: When given any float, it will create a simple ndarray
        where every value is a single number.

    Returns:
        A Measurement object with dummy data for testing. The data will be either all
        a single value for trivial cases, or normalized random data depending on the
        arguements given.
    """
    n_samples = int(sampling_rate * 3600 * duration_hours)
    start_time = datetime(2024, 5, 4, 12, 0, 0)
    delta = timedelta(seconds=1 / sampling_rate)

    time_data = [start_time + i * delta for i in range(n_samples)]

    if all_same_num is not None:
        data = np.full((n_samples, 3), (all_same_num))
        return models.Measurement(
            measurements=data, time=pl.Series(time_data).alias("time")
        )
    else:
        data = np.random.randn(n_samples, 3) - 0.5
        norms = np.linalg.norm(data)
        unit_sphere = data / norms
        return models.Measurement(
            measurements=unit_sphere, time=pl.Series(time_data).alias("time")
        )


def test_get_sampling_rate() -> None:
    """Test get sampling rate."""
    dummy_measure = create_dummy_measurement(
        sampling_rate=60, duration_hours=1, all_same_num=0.0
    )
    calibrator = calibration.GgirCalibration()

    sampling_rate = calibrator._get_sampling_rate(dummy_measure.time)

    assert sampling_rate == 60


def test_no_motion_error() -> None:
    """Test error where no "no motion" epochs found."""
    dummy_measure = create_dummy_measurement(sampling_rate=1, duration_hours=1)

    with pytest.raises(exceptions.NoMotionError):
        calibration._extract_no_motion(
            acceleration=dummy_measure, no_motion_threshold=0.001
        )


def test_extract_no_motion() -> None:
    """Test successful extract no motion."""
    dummy_measure = create_dummy_measurement(
        sampling_rate=1, duration_hours=1, all_same_num=0.32
    )

    no_motion_data = calibration._extract_no_motion(dummy_measure)

    assert np.any(no_motion_data), "No non-motion epochs found"


def test_sphere_error() -> None:
    """Test sphere criteria check in closest point fit."""
    dummy_data = np.full((100, 3), 100)

    calibrator = calibration.GgirCalibration(min_acceleration=0)

    with pytest.raises(exceptions.SphereCriteriaError):
        calibrator._closest_point_fit(dummy_data)


def test_zero_scale_error() -> None:
    """Test error due to scale becomeing zero values."""
    data = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [-0.001, -0.001, -0.001]])

    calibrator = calibration.GgirCalibration(min_acceleration=0)

    with pytest.raises(exceptions.CalibrationError):
        calibrator._closest_point_fit(data)


def test_closest_point_fit() -> None:
    """Test closest point fit."""
    scale = np.array([1.1, 0.9, 0.6])
    offset = np.array([0.1, 0.2, 0.1])
    expected_scale = 1 / scale
    expected_offset = -offset / scale
    data = np.random.randn(1000, 3) - 0.5
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    unit_sphere = data / norms
    calibrator = calibration.GgirCalibration()

    linear_transform = calibrator._closest_point_fit((unit_sphere * scale) + offset)

    assert np.allclose(
        linear_transform.scale, expected_scale, atol=1e-3
    ), f"Scale is {linear_transform.scale} expected {expected_scale}"
    assert np.allclose(
        linear_transform.offset, expected_offset, atol=1e-3
    ), f"Offset is {linear_transform.offset} expected {expected_offset})"


def test_closest_point_fit_constrainedmin() -> None:
    """Test closest point fit."""
    scale = np.array([1.1, 0.9, 0.6])
    offset = np.array([0.1, 0.2, 0.1])
    expected_scale = 1 / scale
    expected_offset = -offset / scale
    data = np.random.randn(1000, 3)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    unit_sphere = data / norms
    calibrator = calibration.ConstrainedMinimizationCalibration()

    linear_transform = calibrator._closest_point_fit((unit_sphere * scale) + offset)

    assert np.allclose(
        linear_transform.scale, expected_scale, atol=1e-3
    ), f"Scale is {linear_transform.scale} expected {expected_scale}"
    assert np.allclose(
        linear_transform.offset, expected_offset, atol=1e-3
    ), f"Offset is {linear_transform.offset} expected {expected_offset})"


def test_closest_point_fit_constrainmin_calibration_error() -> None:
    """Test closest point fit raises CalibrationError for constrained min."""
    data = np.random.randn(1000, 3)
    calibrator = calibration.ConstrainedMinimizationCalibration(max_iterations=0)

    with pytest.raises(exceptions.CalibrationError, match="Optimization failed."):
        calibrator._closest_point_fit(data)


def test_closest_point_fit_constrainmin_calibration_error_minimum() -> None:
    """Test closest point fit raises CalibrationError for constrained min."""
    data = np.random.randn(1000, 3)
    scale = np.array([1.1, 0.9, 0.6])
    calibrator = calibration.ConstrainedMinimizationCalibration(max_calibration_error=0)

    with pytest.raises(
        exceptions.CalibrationError,
        match="Calibration error could not be sufficiently minimized.",
    ):
        calibrator._closest_point_fit(data * scale)


def test_calibrate_calibration_error() -> None:
    """Test error when calibration was not possible."""
    dummy_measure = create_dummy_measurement(
        sampling_rate=1,
        duration_hours=1,
    )
    calibrator = calibration.GgirCalibration(
        max_calibration_error=0.0001, max_iterations=5, min_acceleration=0
    )

    with pytest.raises(exceptions.CalibrationError):
        calibrator._calibrate(dummy_measure)


def test_ggir_calibration_successful() -> None:
    """Test successful ggir calibration."""
    scale = np.array([1.1, 1.01, 0.9])
    offset = np.array([0.1, 0.2, 0.1])
    expected_scale = 1 / scale
    expected_offset = -offset / scale
    dummy_no_motion = np.random.randn(1000, 3)
    norms = np.linalg.norm(dummy_no_motion, axis=1, keepdims=True)
    unit_sphere = dummy_no_motion / norms
    test_data = np.repeat(unit_sphere, repeats=10, axis=0)
    start_time = datetime(2024, 5, 4, 12, 0, 0)
    delta = timedelta(seconds=1)
    time_data = [start_time + i * delta for i in range(10000)]
    dummy_measure = models.Measurement(
        measurements=(test_data * scale) + offset,
        time=pl.Series(time_data).alias("time"),
    )
    calibrator = calibration.GgirCalibration(no_motion_threshold=9999)

    linear_transform = calibrator._calibrate(dummy_measure)

    assert np.allclose(
        linear_transform.scale, expected_scale, atol=1e-3
    ), f"Scale is {linear_transform.scale} expected {expected_scale}"
    assert np.allclose(
        linear_transform.offset, expected_offset, atol=1e-3
    ), f"Offset is {linear_transform.offset} expected {expected_offset}"


@pytest.mark.parametrize(
    "duration_hours, expected_hours",
    [(72, np.array([72])), (84, np.array([72, 84])), (90, np.array([72, 84, 90]))],
)
def test_get_chunk(duration_hours: int, expected_hours: np.ndarray) -> None:
    """Test get chunk."""
    dummy_measure = create_dummy_measurement(
        sampling_rate=1, duration_hours=duration_hours
    )
    expected_lengths = expected_hours * 3600
    dummy_calibrator = calibration.GgirCalibration()

    generator_chunks = dummy_calibrator._get_chunk(acceleration=dummy_measure)

    for i, chunk in enumerate(list(generator_chunks)):
        assert len(chunk.measurements) == expected_lengths[i]
        assert len(chunk.time) == expected_lengths[i]


def test_chunked_calibration_error() -> None:
    """Testing chunked calibration failing after using all chunks."""
    dummy_measure = create_dummy_measurement(sampling_rate=1, duration_hours=84)
    calibrator = calibration.GgirCalibration(
        no_motion_threshold=9999, max_calibration_error=0, chunked=True
    )

    with pytest.raises(
        exceptions.CalibrationError,
        match="After all chunks of data used calibration has failed.",
    ):
        calibrator._chunked_calibration(dummy_measure)


def test_run_hours_calibration_error() -> None:
    """Test error when not enough hours of data."""
    dummy_measure = create_dummy_measurement(sampling_rate=60, duration_hours=10)
    calibrator = calibration.GgirCalibration(min_calibration_hours=72)

    with pytest.raises(exceptions.CalibrationError):
        calibrator.run_calibration(dummy_measure)


def test_run_ggircalibration() -> None:
    """Testing run function when chunked = False."""
    scale = np.array([1.1, 1.01, 0.9])
    offset = np.array([0.1, 0.2, 0.1])
    dummy_no_motion = np.random.randn(1000, 3) - 0.5
    norms = np.linalg.norm(dummy_no_motion, axis=1, keepdims=True)
    unit_sphere = dummy_no_motion / norms
    test_data = np.repeat(unit_sphere, repeats=10, axis=0)
    start_time = datetime(2024, 5, 4, 12, 0, 0)
    delta = timedelta(seconds=1)
    time_data = [start_time + i * delta for i in range(10000)]
    expected_data = models.Measurement(
        measurements=test_data, time=pl.Series(time_data).alias("time")
    )
    dummy_measure = models.Measurement(
        measurements=(test_data * scale) + offset,
        time=pl.Series(time_data).alias("time"),
    )
    calibrator = calibration.GgirCalibration(
        no_motion_threshold=9999, min_calibration_hours=1
    )

    result = calibrator.run_calibration(dummy_measure)

    assert isinstance(
        result, models.Measurement
    ), f"was expecting type models.Measurement, object is of type {type(result)}"
    assert np.allclose(
        result.measurements, expected_data.measurements, atol=1e-3
    ), "Measurement data did not match"
    assert result.time.equals(expected_data.time), "Time series are not equal"


def test_run_constrainedmincalibration() -> None:
    """Testing run calibration function for the constrained minimization."""
    scale = np.array([1.1, 1.01, 0.9])
    offset = np.array([0.1, 0.2, 0.1])
    dummy_no_motion = np.random.randn(1000, 3) - 0.2
    norms = np.linalg.norm(dummy_no_motion, axis=1, keepdims=True)
    unit_sphere = dummy_no_motion / norms
    test_data = np.repeat(unit_sphere, repeats=10, axis=0)
    start_time = datetime(2024, 5, 4, 12, 0, 0)
    delta = timedelta(seconds=1)
    time_data = [start_time + i * delta for i in range(10000)]
    expected_data = models.Measurement(
        measurements=test_data, time=pl.Series(time_data).alias("time")
    )
    dummy_measure = models.Measurement(
        measurements=(test_data * scale) + offset,
        time=pl.Series(time_data).alias("time"),
    )
    calibrator = calibration.ConstrainedMinimizationCalibration(
        no_motion_threshold=9999
    )

    result = calibrator.run_calibration(dummy_measure)

    assert isinstance(
        result, models.Measurement
    ), f"was expecting type models.Measurement, object is of type {type(result)}"
    assert np.allclose(
        result.measurements, expected_data.measurements, atol=1e-3
    ), "Measurement data did not match"
    assert result.time.equals(expected_data.time), "Time series are not equal"


def test_run_chunked_calibration() -> None:
    """Testing run function when chunked = True."""
    scale = np.array([1.1, 1.01, 0.9])
    offset = np.array([0.1, 0.2, 0.1])
    dummy_no_motion = np.random.randn(1000, 3) - 0.5
    norms = np.linalg.norm(dummy_no_motion, axis=1, keepdims=True)
    unit_sphere = dummy_no_motion / norms
    test_data = np.repeat(unit_sphere, repeats=10, axis=0)
    start_time = datetime(2024, 5, 4, 12, 0, 0)
    delta = timedelta(seconds=1)
    time_data = [start_time + i * delta for i in range(10000)]
    expected_data = models.Measurement(
        measurements=test_data, time=pl.Series(time_data).alias("time")
    )
    dummy_measure = models.Measurement(
        measurements=(test_data * scale) + offset,
        time=pl.Series(time_data).alias("time"),
    )
    calibrator = calibration.GgirCalibration(
        no_motion_threshold=9999, chunked=True, min_calibration_hours=1
    )

    result = calibrator.run_calibration(dummy_measure)

    assert isinstance(
        result, models.Measurement
    ), f"was expecting type models.Measurement, object is of type {type(result)}"
    assert np.allclose(
        result.measurements, expected_data.measurements, atol=1e-3
    ), "Measurement data did not match"
    assert result.time.equals(expected_data.time), "Time series are not equal"


def test_calibration_dispatcher_bad() -> None:
    """Test calibration dispatcher with bad calibrator."""
    dummy_measure = create_dummy_measurement(sampling_rate=1, duration_hours=84)
    bad_calibrator_name = "bad"

    with pytest.raises(ValueError, match="Unknown calibrator."):
        calibration.CalibrationDispatcher(bad_calibrator_name).run(dummy_measure)  # type: ignore[arg-type] # failing on purpose due to bad calibrator name
