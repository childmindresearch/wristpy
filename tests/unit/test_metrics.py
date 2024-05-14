"""Testing functions of metrics module."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from wristpy.core import models
from wristpy.processing import metrics

TEST_LENGTH = 100


@pytest.mark.parametrize(
    "x,y,z, expected_enmo",
    [
        (np.zeros(TEST_LENGTH), np.zeros(TEST_LENGTH), np.zeros(TEST_LENGTH), 0),
        (np.ones(TEST_LENGTH), np.zeros(TEST_LENGTH), np.zeros(TEST_LENGTH), 0),
        (np.ones(TEST_LENGTH) * 2, np.zeros(TEST_LENGTH), np.zeros(TEST_LENGTH), 1),
        (np.ones(TEST_LENGTH), np.ones(TEST_LENGTH) * 2, np.ones(TEST_LENGTH) * 2, 2),
    ],
)
def test_euclidean_norm_minus_one(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, expected_enmo: float
) -> None:
    """Tests the euclidean norm function."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime = pl.Series("time", [dummy_date] * TEST_LENGTH)
    test_acceleration = models.Measurement(
        measurements=np.column_stack((x, y, z)), time=dummy_datetime
    )

    enmo_results = metrics.euclidean_norm_minus_one(test_acceleration)

    assert np.all(
        np.isclose(enmo_results.measurements, expected_enmo)
    ), f"Expected {expected_enmo}"

    assert enmo_results.time.equals(
        test_acceleration.time
    ), "Input time attribute does not match output time attribute."

    assert enmo_results.measurements.shape == (
        TEST_LENGTH,
    ), f"Expected enmo shape: ({TEST_LENGTH},), got ({enmo_results.measurements.shape})"


@pytest.mark.parametrize(
    "measurements, expected_anglez",
    [
        (np.array([[0, 0, 0]]), np.nan),
        (np.array([[1, 0, 0]]), 0),
        (np.array([[0, 0, 1]]), 90),
        (np.array([[0, 0, -1]]), -90),
        (np.array([[1, 1, 1]]), 35.264389),
    ],
)
def test_angle_relative_to_horizontal(
    measurements: np.ndarray, expected_anglez: int
) -> None:
    """Test angle relative to hroizontal function."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime = pl.Series("time", [dummy_date])
    test_acceleration = models.Measurement(
        measurements=measurements, time=dummy_datetime
    )

    angle_z_results = metrics.angle_relative_to_horizontal(test_acceleration)

    assert np.all(
        np.isclose(angle_z_results.measurements, expected_anglez, equal_nan=True)
    ), f"Expected {expected_anglez}, got: {angle_z_results.measurements}"

    assert angle_z_results.time.equals(
        test_acceleration.time
    ), "Input time attribute does not match output time attribute."

    assert angle_z_results.measurements.shape == (1,), (
        f"Expected anglez shape: {(1,)}," f"got({angle_z_results.measurements.shape})"
    )


def test_rolling_median_wrong_ndims() -> None:
    """Test raising value error when ndims is 1 or fewer."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime = pl.Series("time", [dummy_date])
    bad_array = np.array([1, 2, 3])
    test_measurement = models.Measurement(measurements=bad_array, time=dummy_datetime)

    with pytest.raises(ValueError):
        metrics.rolling_median(test_measurement)


def test_rolling_median_window_size() -> None:
    """Test the window size being <=1."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime = pl.Series("time", [dummy_date])
    test_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    test_measurement = models.Measurement(measurements=test_array, time=dummy_datetime)

    with pytest.raises(ValueError):
        metrics.rolling_median(test_measurement, window_size=0)


@pytest.mark.parametrize(
    "window_size, expected_output",
    [
        (3, np.array([[5, 3.5, 2], [4, 5, 1], [6.5, 6.5, 1]])),
        (2, np.array([[5, 3.5, 2], [4, 5, 1], [6.5, 6.5, 1]])),
    ],
)
def test_rolling_median(window_size: int, expected_output: np.ndarray) -> np.ndarray:
    """Testing proper function of rolling median function."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime = pl.Series("time", [dummy_date] * 3)
    test_matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [9.0, 5.0, 1.0],
            [4.0, 8.0, 1.0],
        ]
    )

    test_measurement = models.Measurement(measurements=test_matrix, time=dummy_datetime)

    test_result = metrics.rolling_median(test_measurement, window_size=window_size)

    assert test_result.measurements.shape == expected_output.shape, (
        f"measurements array are not the same shape. Expected {expected_output.shape}, "
        f"instead got: {test_result.measurements.shape}"
    )
    assert np.all(
        np.isclose(test_result.measurements, expected_output)
    ), "test_result values and expected_output values are not the same."
