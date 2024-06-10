"""Testing functions of metrics module."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from wristpy.core import models
from wristpy.processing import metrics

TEST_LENGTH = 100


@pytest.fixture(scope="module")
def create_acceleration() -> pl.DataFrame:
    """Fixture to create a dummy acceleration DataFrame to be used in multiple tests."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + timedelta(seconds=i) for i in range(10000)]
    test_time = pl.Series("time", dummy_datetime_list)
    acceleration_polars_df = pl.DataFrame(
        {
            "X": np.ones(10000),
            "Y": np.ones(10000),
            "Z": np.ones(10000),
            "time": test_time,
        }
    )
    return acceleration_polars_df


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


@pytest.mark.parametrize(
    "modifier, expected_result",
    [
        (1, 1),
        (0, 0),
    ],
)
def test_compute_nonwear_value_per_axis(
    create_acceleration: pl.DataFrame, modifier: int, expected_result: int
) -> None:
    """Test the nonwear value per axis function."""
    std_criteria = modifier
    range_criteria = modifier
    acceleration = create_acceleration
    acceleration = acceleration.with_columns(pl.col("time").set_sorted())
    acceleration_grouped = acceleration.group_by_dynamic(
        index_column="time", every="5s"
    ).agg([pl.all().exclude(["time"])])

    test_resultx = metrics._compute_nonwear_value_per_axis(
        acceleration_grouped["X"], std_criteria, range_criteria
    )
    test_resulty = metrics._compute_nonwear_value_per_axis(
        acceleration_grouped["Y"], std_criteria, range_criteria
    )
    test_resultz = metrics._compute_nonwear_value_per_axis(
        acceleration_grouped["Z"], std_criteria, range_criteria
    )

    assert (
        test_resultx == expected_result
    ), f"Expected {expected_result}, got: {test_resultx}"
    assert (
        test_resulty == expected_result
    ), f"Expected {expected_result}, got: {test_resulty}"
    assert (
        test_resultz == expected_result
    ), f"Expected {expected_result}, got: {test_resultz}"


@pytest.mark.parametrize(
    "modifier, expected_result",
    [
        (1, 1),
        (0, 0),
    ],
)
def test_detect_nonwear(
    create_acceleration: pl.DataFrame, modifier: int, expected_result: int
) -> None:
    """Test the detect nonwear function."""
    short_epoch_length = 5
    long_epoch_length = 20
    std_criteria = modifier
    range_criteria = modifier
    acceleration_df = create_acceleration
    acceleration = models.Measurement(
        measurements=acceleration_df.select(["X", "Y", "Z"]).to_numpy(),
        time=acceleration_df["time"],
    )
    expected_time_length = math.ceil(len(acceleration.time) / short_epoch_length)

    test_result = metrics.detect_nonwear(
        acceleration,
        short_epoch_length,
        long_epoch_length,
        std_criteria,
        range_criteria,
    )

    assert np.all(
        test_result.measurements == modifier
    ), f"Expected non-wear flag value to be {expected_result}, got: {test_result}"
    assert (
        len(test_result.time) == expected_time_length
    ), f"Expected time to be {expected_time_length}, got: {len(test_result.time)}"
