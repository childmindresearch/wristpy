"""test the calibration module."""

from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import polars as pl
import pytest

from wristpy.core import models
from wristpy.processing import calibration


def create_dummy_measurement(
    sampling_rate: int,
    duration_hours: float = 84.0,
    loc: float = 1.0,
    scale: float = 0.1,
    all_same_num: float | None = None,
) -> models.Measurement:
    """Create dummy measurement."""
    n_samples = int(sampling_rate * 3600 * duration_hours)
    start_time = datetime.now()
    delta = timedelta(seconds=1 / sampling_rate)

    time_data = [start_time + i * delta for i in range(n_samples)]

    if all_same_num is not None:
        measurement_data = np.full((n_samples, 3), (all_same_num))
    else:
        measurement_data = np.random.normal(loc=loc, scale=scale, size=(n_samples, 3))

    dummy_measurement = models.Measurement(
        measurements=measurement_data, time=pl.Series(time_data).alias("time")
    )

    return dummy_measurement


def test_get_sampling_rate() -> None:
    """Find sampling rate."""
    dummy_measure = create_dummy_measurement(
        sampling_rate=60, duration_hours=1, all_same_num=0.0
    )
    calibrator = calibration.Calibration()

    sampling_rate = calibrator._get_sampling_rate(dummy_measure)

    assert sampling_rate == 60


def test_no_motion_error() -> None:
    """Check extract no motion."""
    date = datetime.now()
    delta = timedelta(seconds=1)
    dummy_datetime = [date + (i * delta) for i in range(60)]
    dummy_data = np.random.uniform(-0.29, 0.29, (60, 3))
    dummy_measure = models.Measurement(
        measurements=dummy_data, time=pl.Series(dummy_datetime).alias("time")
    )
    calibrator = calibration.Calibration()

    with pytest.raises(
        calibration.NoMotionError,
        match="Zero non-motion epochs found. Data did not meet criteria.",
    ):
        calibrator._extract_no_motion(acceleration=dummy_measure)


def test_extract_no_motion_sphere_error() -> None:
    """Check extract no motion."""
    dummy_measure = create_dummy_measurement(
        sampling_rate=1, duration_hours=1, all_same_num=0.29
    )
    calibrator = calibration.Calibration()

    with pytest.raises(calibration.SphereCriteriaError):
        calibrator._extract_no_motion(acceleration=dummy_measure)


def test_extract_no_motion() -> None:
    """Check extract no motion."""
    dummy_measure = create_dummy_measurement(
        sampling_rate=1, duration_hours=1, all_same_num=0.32
    )

    dummy_measure.measurements[30:] *= -1

    calibrator = calibration.Calibration()

    no_motion_data = calibrator._extract_no_motion(dummy_measure)

    assert no_motion_data.shape[0] > 0, "No non-motion epochs found"
    assert np.all(no_motion_data.min(axis=0) <= -calibrator.min_acceleration)
    assert np.all(no_motion_data.max(axis=0) >= calibrator.min_acceleration)


@pytest.mark.parametrize(
    "scale, offset",
    [(1.001, 0.001), (1.0, 0.0)],
)
def test_closest_point_fit(scale: float, offset: float) -> None:
    """Test closest point fit."""
    dummy_no_motion = np.random.normal(size=(100, 3))

    norms = np.linalg.norm(dummy_no_motion, axis=1, keepdims=True)
    unit_sphere = dummy_no_motion / norms
    calibrator = calibration.Calibration()

    linear_transform = calibrator._closest_point_fit((unit_sphere * scale) + offset)

    assert np.allclose(
        np.mean(linear_transform.scale), 1 / scale, atol=1e-4
    ), f"Scale is {np.mean(linear_transform.scale)} expected {1/scale}"
    assert np.allclose(
        np.mean(linear_transform.offset), -offset / scale, atol=1e-4
    ), f"Offset is {np.mean(linear_transform.offset)} expected {-offset/scale}"


def test_calibrate_calibration_error() -> None:
    """Test failure due to calibration error."""
    dummy_measure = create_dummy_measurement(
        sampling_rate=1, duration_hours=1, loc=0, scale=1
    )
    calibrator = calibration.Calibration(
        min_calibration_error=0.0001, max_iterations=5, min_standard_deviation=2.0
    )

    with pytest.raises(calibration.CalibrationError):
        calibrator._calibrate(dummy_measure)


@pytest.mark.parametrize(
    "scale, offset",
    [(1.1, 0.01)],
)
def test_calibration_successful(scale: float, offset: float) -> None:
    """Test successful calibration."""
    dummy_measure = create_dummy_measurement(
        sampling_rate=1, duration_hours=1, all_same_num=1 / np.sqrt(3)
    )
    dummy_measure.measurements[1800:] *= -1
    dummy_measure.measurements = (dummy_measure.measurements * scale) + offset
    expected_scale = 1 / scale
    expected_offset = -offset
    calibrator = calibration.Calibration(min_acceleration=0)

    linear_transform = calibrator._calibrate(dummy_measure)

    assert np.allclose(
        np.mean(linear_transform.scale), expected_scale, atol=1e-3
    ), f"Scale is {np.mean(linear_transform.scale)} expected {expected_scale}"
    assert np.allclose(
        np.mean(linear_transform.offset), expected_offset, atol=1e-3
    ), f"Offset is {np.mean(linear_transform.offset)} expected {-expected_offset}"


@pytest.mark.parametrize(
    "duration_hours, expected_hours",
    [(72, np.array([72])), (84, np.array([72, 84])), (90, np.array([72, 84, 90]))],
)
def test_get_chunk(duration_hours: int, expected_hours: np.ndarray) -> None:
    """Test take chunk."""
    dummy_measure = create_dummy_measurement(
        sampling_rate=1, duration_hours=duration_hours
    )
    expected_lengths = expected_hours * 3600
    dummy_calibrator = calibration.Calibration()

    generator_chunks = dummy_calibrator._get_chunk(acceleration=dummy_measure)

    for i, chunk in enumerate(list(generator_chunks)):
        assert len(chunk.measurements) == expected_lengths[i]
        assert len(chunk.time) == expected_lengths[i]


def test_run_hours_value_error() -> None:
    """Test error when not enough hours of data."""
    dummy_measure = create_dummy_measurement(sampling_rate=60, duration_hours=10)
    calibrator = calibration.Calibration(min_calibration_hours=72)

    with pytest.raises(ValueError):
        calibrator.run(dummy_measure)
