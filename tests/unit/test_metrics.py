"""Testing functions of metrics module."""

import math
import pathlib
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from wristpy.io.readers import readers
from wristpy.core import models
from wristpy.processing import metrics

TEST_LENGTH = 100


@pytest.fixture
def create_acceleration() -> pl.DataFrame:
    """Fixture to create a dummy acceleration DataFrame to be used in multiple tests."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + timedelta(seconds=i) for i in range(1000)]
    test_time = pl.Series("time", dummy_datetime_list)
    acceleration_polars_df = pl.DataFrame(
        {
            "X": np.ones(1000),
            "Y": np.ones(1000),
            "Z": np.ones(1000),
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
    dummy_datetime_list = [
        dummy_date + timedelta(seconds=i) for i in range(TEST_LENGTH)
    ]
    test_time = pl.Series("time", dummy_datetime_list)
    test_acceleration = models.Measurement(
        measurements=np.column_stack((x, y, z)), time=test_time
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
    "nonwear_value_array, expected_result",
    [
        (
            np.array([0, 1, 1, 0, 1, 1, 1]),
            np.array([0, 1, 1, 0, 1, 1, 1]),
        ),
        (
            np.array([0, 1, 1, 2, 1, 2, 1, 3]),
            np.array([0, 1, 1, 2, 2, 2, 2, 3]),
        ),
    ],
)
def test_cleanup_isolated_ones_nonwear_value(
    nonwear_value_array: np.ndarray, expected_result: np.ndarray
) -> None:
    """Test the cleanup isolated ones nonwear value function."""
    test_result = metrics._cleanup_isolated_ones_nonwear_value(nonwear_value_array)

    assert np.all(
        test_result == expected_result
    ), f"Expected {expected_result}, got: {test_result}"


def test_group_acceleration_data_by_time() -> None:
    """Test the group acceleration data by time function."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + timedelta(seconds=i) for i in range(1001)]
    test_time = pl.Series(dummy_datetime_list)
    measurements = np.ones((1001, 3))
    acceleration = models.Measurement(measurements=measurements, time=test_time)
    window_length = int(10)
    expected_time_length = math.ceil(len(test_time) / window_length)
    expected_result_shape = acceleration.measurements.shape[1] + 1

    result = metrics._group_acceleration_data_by_time(acceleration, window_length)

    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == expected_time_length
    assert result.shape[1] == expected_result_shape


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
    acceleration = create_acceleration.with_columns(pl.col("time").set_sorted())
    acceleration_grouped = acceleration.group_by_dynamic(
        index_column="time", every="5s"
    ).agg([pl.all().exclude(["time"])])

    test_resultx = metrics._compute_nonwear_value_per_axis(
        acceleration_grouped["X"], std_criteria=modifier
    )

    assert (
        test_resultx == expected_result
    ), f"Expected {expected_result}, got: {test_resultx}"


def test_compute_nonwear_value_array(create_acceleration: pl.DataFrame) -> None:
    """Test the compute nonwear value array function."""
    n_short_epoch_in_long_epoch = int(4)
    create_acceleration = create_acceleration.with_columns(pl.col("time").set_sorted())
    acceleration_grouped = create_acceleration.group_by_dynamic(
        index_column="time", every="5s"
    ).agg([pl.all().exclude(["time"])])
    expected_time_length = len(acceleration_grouped)
    expected_result = 3

    test_result = metrics._compute_nonwear_value_array(
        acceleration_grouped,
        n_short_epoch_in_long_epoch,
        std_criteria=1,
    )

    assert np.all(
        test_result == expected_result
    ), f"Expected {expected_result}, got: {test_result}"
    assert (
        len(test_result) == expected_time_length
    ), f"Expected time to be {expected_time_length}, got: {len(test_result)}"


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
    n_short_epoch_in_long_epoch = int(4)
    acceleration_df = create_acceleration
    acceleration = models.Measurement(
        measurements=acceleration_df.select(["X", "Y", "Z"]).to_numpy(),
        time=acceleration_df["time"],
    )
    expected_time_length = math.ceil(len(acceleration.time) / short_epoch_length)

    test_result = metrics.detect_nonwear(
        acceleration,
        short_epoch_length,
        n_short_epoch_in_long_epoch,
        std_criteria=modifier,
    )

    assert np.all(
        test_result.measurements == modifier
    ), f"Expected non-wear flag value to be {expected_result}, got: {test_result}"
    assert (
        len(test_result.time) == expected_time_length
    ), f"Expected time to be {expected_time_length}, got: {len(test_result.time)}"


def test_interpolate_time(
    sample_data_gt3x: pathlib.Path,
    actigraph_interpolation_r_version: pathlib.Path,
) -> None:
    """Test the time values for the  interpolate function."""
    expected_data = pl.read_csv(actigraph_interpolation_r_version)
    expected_time = expected_data["time"].str.strptime(
        pl.Datetime("ns"), format="%Y-%m-%d %H:%M:%S%.f"
    )
    expected_ms = expected_time.dt.epoch(time_unit="ms").to_numpy()
    test_data = readers.read_watch_data(sample_data_gt3x)

    interpolated_acceleration = metrics.interpolate_measure(
        acceleration=test_data.acceleration, new_frequency=100
    )
    interpolated_ms = interpolated_acceleration.time.dt.epoch(time_unit="ms").to_numpy()

    assert len(expected_time) == len(
        interpolated_acceleration.time
    ), "Timestamp series are not the same length."
    assert np.allclose(
        expected_ms, interpolated_ms, atol=1
    ), "Timestamps don't match within tolerance. "


def test_interpolate_data(
    sample_data_gt3x: pathlib.Path, actigraph_interpolation_r_version: pathlib.Path
) -> None:
    """Test the acceleration data from the interpolate."""
    expected_data = pl.read_csv(actigraph_interpolation_r_version)
    expected_acceleration = expected_data["X", "Y", "Z"].to_numpy()
    test_data = readers.read_watch_data(sample_data_gt3x)

    interpolated_acceleration = metrics.interpolate_measure(
        acceleration=test_data.acceleration, new_frequency=100
    )

    assert (
        expected_acceleration.shape == interpolated_acceleration.measurements.shape
    ), "Shape error."
    for axis in range(3):
        correlation = np.corrcoef(
            expected_acceleration.T[axis, :],
            interpolated_acceleration.measurements.T[axis, :],
        )
        assert np.all(
            correlation > 0.99
        ), f"Axis:{axis} did not meet the threshold, current values: {correlation}"
