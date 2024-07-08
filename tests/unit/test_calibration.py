"""test the calibration module."""

from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import polars as pl
import pytest

from wristpy.core import models
from wristpy.processing import calibration


def create_dummy_measurement(
    sampling_rate: int, duration_hours: float = 84.0, all_same_num: float | None = None
) -> models.Measurement:
    """Create dummy measurement."""
    n_samples = int(sampling_rate * 3600 * duration_hours)
    start_time = datetime.now()
    delta = timedelta(seconds=1 / sampling_rate)

    time_data = [start_time + i * delta for i in range(n_samples)]

    if all_same_num is not None:
        measurement_data = np.full((n_samples, 3), (all_same_num))
    else:
        measurement_data = np.random.normal(loc=1.0, scale=0.1, size=(n_samples, 3))

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


def test_extract_no_motion_value_error() -> None:
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
        ValueError, match="Zero non-motion epochs found. Data did not meet criteria."
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
        sampling_rate=1, duration_hours=(1 / 60), all_same_num=0.31
    )
    dummy_measure.measurements[30:] *= -1

    calibrator = calibration.Calibration()

    no_motion_data = calibrator._extract_no_motion(dummy_measure)

    assert no_motion_data.shape[0] > 0, "No non-motion epochs found"
    assert np.all(no_motion_data.min(axis=0) <= -calibrator.min_acceleration)
    assert np.all(no_motion_data.max(axis=0) >= calibrator.min_acceleration)


@pytest.mark.parametrize(
    "scale, offset, expected_scale, expected_offset",
    [(1.01, 0.001, 1 / 1.01, -0.001 / 1.01), (1.0, 0.0, 1.0, 0.0)],
)
def test_closest_point_fit(
    scale: float, offset: float, expected_scale: float, expected_offset: float
) -> None:
    """Test closest point fit."""
    dummy_no_motion = np.random.normal(size=(100, 3))

    norms = np.linalg.norm(dummy_no_motion, axis=1, keepdims=True)
    unit_sphere = dummy_no_motion / norms
    calibrator = calibration.Calibration()

    linear_transform = calibrator._closest_point_fit((unit_sphere * scale) + offset)

    assert np.allclose(
        np.mean(linear_transform["scale"]), expected_scale, atol=1e-4
    ), f'Scale is {np.mean(linear_transform["scale"])} expected {expected_scale}'
    assert np.allclose(
        np.mean(linear_transform["offset"]), expected_offset, atol=1e-4
    ), f'Offset is {np.mean(linear_transform["offset"])} expected {expected_offset}'


def test_calibrate_calibration_error() -> None:
    """Test failure due to calibration error."""
    no_motion_data = np.full((3600, 3), 1.0)
    dummy_measure = create_dummy_measurement(
        sampling_rate=1, duration_hours=1, all_same_num=1.0
    )
    calibrator = calibration.Calibration(min_calibration_error=0.0001, max_iterations=5)

    with mock.patch.object(
        calibration.Calibration, "_extract_no_motion"
    ) as mock_extracted_data:
        mock_extracted_data.return_value = no_motion_data

        with pytest.raises(calibration.CalibrationError):
            calibrator._calibrate(dummy_measure)


def test_calibration_successful() -> None:
    """Test successful calibration."""
    no_motion_data = np.full((3600, 3), (1 / np.sqrt(3) + 1))
    linear_transformation = {"scale": 1, "offset": -1}
    dummy_measure = create_dummy_measurement(
        sampling_rate=1, duration_hours=1, all_same_num=(1 / np.sqrt(3) + 1)
    )

    with mock.patch.object(
        calibration.Calibration, "_extract_no_motion"
    ) as mock_extracted_data:
        with mock.patch.object(
            calibration.Calibration, "_closest_point_fit"
        ) as mock_closest_point_fit:
            mock_extracted_data.return_value = no_motion_data
            mock_closest_point_fit.return_value = linear_transformation
            calibrator = calibration.Calibration()

            result = calibrator._calibrate(dummy_measure)

            assert result["scale"] == linear_transformation["scale"]
            assert result["offset"] == linear_transformation["offset"]


@pytest.mark.parametrize(
    "chunk_num, duration_hours",
    [(0, (72)), ((1), (84)), ((2), (90)), ((2), (96))],
)
def test_take_chunk(chunk_num: int, duration_hours: int) -> None:
    """Test take chunk."""
    expected_len = 3600 * duration_hours
    dummy_measure = create_dummy_measurement(
        sampling_rate=1, duration_hours=duration_hours
    )
    dummy_calibrator = calibration.Calibration()

    subset = dummy_calibrator._take_chunk(
        acceleration=dummy_measure, chunk_num=chunk_num
    )

    assert len(subset.measurements) == expected_len
    assert len(subset.time) == expected_len


def test_chunked_calibration_fail() -> None:
    """Test chunked calibration failure."""
    dummy_measurement = create_dummy_measurement(sampling_rate=1, duration_hours=84)

    calibrator = calibration.Calibration(
        chunked=True,
        min_calibration_hours=72,
        max_iterations=5,
        min_calibration_error=0.00001,
    )

    with pytest.raises(calibration.CalibrationError):
        calibrator.run(dummy_measurement)


def test_chunked_calibration_fail_and_succeed() -> None:
    """Test chunked calibration failure, then success after taking a chunk."""
    dummy_measurement = create_dummy_measurement(sampling_rate=1, duration_hours=84)
    mock_linear_transformation = {
        "scale": np.array([1, 1, 1]),
        "offset": np.array([0, 0, 0]),
    }

    calibrator = calibration.Calibration(chunked=True, min_calibration_hours=72)

    with mock.patch.object(calibration.Calibration, "_calibrate") as mock_calibrate:
        mock_calibrate.side_effect = [
            calibration.CalibrationError(),
            mock_linear_transformation,
        ]

        result = calibrator._chunked_calibration(dummy_measurement)

        assert np.allclose(result["scale"], mock_linear_transformation["scale"])
        assert np.allclose(result["offset"], mock_linear_transformation["offset"])
        assert mock_calibrate.call_count == 2

        first_call_samples = mock_calibrate.call_args_list[0][0][0].measurements.shape[
            0
        ]
        second_call_samples = mock_calibrate.call_args_list[1][0][0].measurements.shape[
            0
        ]
        assert first_call_samples == (3600 * 72)
        assert second_call_samples == (3600 * 84)


def test_run_hours_value_error() -> None:
    """Test error when not enough hours of data."""
    dummy_measure = create_dummy_measurement(sampling_rate=60, duration_hours=10)
    calibrator = calibration.Calibration(min_calibration_hours=72)

    with pytest.raises(ValueError):
        calibrator.run(dummy_measure)


@pytest.mark.parametrize(
    "scale, offset",
    [([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]), ([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])],
)
def test_run_successful_calibration(scale: list[float], offset: list[float]) -> None:
    """Test the run method when calibration succeeds."""
    dummy_measure = create_dummy_measurement(sampling_rate=1, duration_hours=24)

    calibrator = calibration.Calibration(min_calibration_hours=12)

    mock_transformation = {
        "scale": np.array(scale),
        "offset": np.array(offset),
    }
    expected_data = (
        dummy_measure.measurements * mock_transformation["scale"]
    ) + mock_transformation["offset"]
    with mock.patch.object(
        calibration.Calibration, "_calibrate", return_value=mock_transformation
    ):
        result = calibrator.run(dummy_measure)

    assert isinstance(result, models.Measurement)
    assert np.allclose(result.measurements, expected_data)


@pytest.mark.parametrize(
    "scale, offset",
    [([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]), ([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])],
)
def test_run_successful_chunked_calibration(
    scale: list[float], offset: list[float]
) -> None:
    """Test chunked calibration when calibration succeeds."""
    dummy_measure = create_dummy_measurement(sampling_rate=1, duration_hours=24)

    calibrator = calibration.Calibration(min_calibration_hours=12, chunked=True)

    mock_transformation = {
        "scale": np.array(scale),
        "offset": np.array(offset),
    }
    expected_data = (
        dummy_measure.measurements * mock_transformation["scale"]
    ) + mock_transformation["offset"]
    with mock.patch.object(
        calibration.Calibration,
        "_chunked_calibration",
        return_value=mock_transformation,
    ):
        result = calibrator.run(dummy_measure)

    assert isinstance(result, models.Measurement)
    assert np.allclose(result.measurements, expected_data)
