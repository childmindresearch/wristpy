"""Testing functions of metrics module."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from wristpy.core.models import Measurement
from wristpy.processing.metrics import (
    angle_relative_to_horizontal,
    euclidean_norm_minus_one,
)

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
    test_acceleration = Measurement(
        measurements=np.column_stack((x, y, z)), time=dummy_datetime
    )

    enmo_results = euclidean_norm_minus_one(test_acceleration)

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
    "x,y,z, expected_anglez",
    [
        (np.zeros(TEST_LENGTH), np.zeros(TEST_LENGTH), np.zeros(TEST_LENGTH), np.nan),
        (np.ones(TEST_LENGTH), np.zeros(TEST_LENGTH), np.zeros(TEST_LENGTH), 0),
        (
            np.ones(TEST_LENGTH) * np.sqrt(3),
            np.zeros(TEST_LENGTH),
            np.ones(TEST_LENGTH),
            30,
        ),
        (np.ones(TEST_LENGTH), np.zeros(TEST_LENGTH), np.ones(TEST_LENGTH), 45),
        (
            np.ones(TEST_LENGTH),
            np.zeros(TEST_LENGTH),
            np.ones(TEST_LENGTH) * np.sqrt(3),
            60,
        ),
        (np.zeros(TEST_LENGTH), np.zeros(TEST_LENGTH), np.ones(TEST_LENGTH), 90),
        (
            np.ones(TEST_LENGTH),
            np.zeros(TEST_LENGTH),
            np.ones(TEST_LENGTH) * -0.57735,
            -30,
        ),
        (
            np.ones(TEST_LENGTH),
            np.zeros(TEST_LENGTH),
            np.ones(TEST_LENGTH) * -(np.sqrt(3)),
            -60,
        ),
        (np.ones(TEST_LENGTH), np.zeros(TEST_LENGTH), np.ones(TEST_LENGTH) * -1, -45),
        (np.zeros(TEST_LENGTH), np.zeros(TEST_LENGTH), np.ones(TEST_LENGTH) * -1, -90),
    ],
)
def test_angle_relative_to_horizontal(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, expected_anglez: int
) -> None:
    """Test angle relative to hroizontal function."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime = pl.Series("time", [dummy_date] * TEST_LENGTH)
    test_acceleration = Measurement(
        measurements=np.column_stack((x, y, z)), time=dummy_datetime
    )

    angle_z_results = angle_relative_to_horizontal(test_acceleration)

    assert np.all(
        np.isclose(angle_z_results.measurements, expected_anglez, equal_nan=True)
    ), f"Expected {expected_anglez}"

    assert angle_z_results.time.equals(
        test_acceleration.time
    ), "Input time attribute does not match output time attribute."

    assert angle_z_results.measurements.shape == (TEST_LENGTH,), (
        f"Expected anglez shape: ({TEST_LENGTH},),"
        f"got({angle_z_results.measurements.shape})"
    )
