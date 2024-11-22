"""Testing functions of metrics module."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

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


@pytest.fixture
def create_temperature() -> pl.DataFrame:
    """Fixture to create a dummy temperature DataFrame to be used in multiple tests."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + timedelta(seconds=i) for i in range(1000)]
    test_time = pl.Series("time", dummy_datetime_list)
    temperature_polars_df = pl.DataFrame(
        {
            "temperature": np.linspace(1.0, 25.0, 1000),
            "time": test_time,
        }
    )
    return temperature_polars_df


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


@pytest.mark.parametrize("temp_length", [498, 500])
def test_pre_proc_temp(create_acceleration: pl.DataFrame, temp_length: int) -> None:
    """Test the pre-process temperature function for time padding."""
    acceleration_df = create_acceleration
    acceleration = models.Measurement(
        measurements=acceleration_df.select(["X", "Y", "Z"]).to_numpy(),
        time=acceleration_df["time"],
    )

    dummy_date = datetime(2024, 5, 2)
    dummy_datetime_list = [
        dummy_date + timedelta(seconds=2 * i) for i in range(temp_length)
    ]
    temp_time = pl.Series("time", dummy_datetime_list).cast(pl.Datetime("ns"))
    temperature = models.Measurement(
        measurements=np.ones(len(temp_time), dtype=np.float32), time=temp_time
    )
    expected_time_length = len(acceleration.time)
    expected_end_time = int(acceleration.time[-1].timestamp())

    low_pass_temp = metrics._pre_process_temperature(temperature, acceleration)

    assert len(low_pass_temp["time"]) == expected_time_length
    assert int(low_pass_temp["time"][-1].timestamp()) == expected_end_time
    assert np.all(low_pass_temp["temperature"].to_numpy() == 1.0)


@pytest.mark.parametrize("std_criteria, expected_result", [(0, 0), (0.013, 1)])
def test_cta_non_wear(
    create_acceleration: pl.DataFrame,
    create_temperature: pl.DataFrame,
    std_criteria: float,
    expected_result: int,
) -> None:
    """Test the cta non wear function."""
    acceleration_df = create_acceleration
    acceleration = models.Measurement(
        measurements=acceleration_df.select(["X", "Y", "Z"]).to_numpy(),
        time=acceleration_df["time"],
    )
    temperature_df = create_temperature
    temperature = models.Measurement(
        measurements=temperature_df["temperature"].to_numpy(),
        time=temperature_df["time"],
    )
    expected_time_length = math.ceil(len(acceleration.time) / 60)

    non_wear_array = metrics.combined_temp_accel_detect_nonwear(
        acceleration, temperature, std_criteria=std_criteria
    )

    assert len(non_wear_array.time) == expected_time_length
    assert np.all(non_wear_array.measurements == expected_result)


def test_cta_non_wear_decreasing_temp() -> None:
    """Test the CTA algorithm when temperature is decreasing."""
    num_samples = int(1000)
    time_list = [
        datetime(2024, 5, 2) + timedelta(milliseconds=100 * i)
        for i in range(num_samples)
    ]
    time = pl.Series("time", time_list, dtype=pl.Datetime("ns"))
    acceleration = models.Measurement(measurements=np.ones((num_samples, 3)), time=time)
    temperature = models.Measurement(
        measurements=np.linspace(20, 25, num_samples)[::-1], time=time
    )

    non_wear_array = metrics.combined_temp_accel_detect_nonwear(
        acceleration, temperature, std_criteria=0
    )

    assert np.all(non_wear_array.measurements[1:] == 1)


def test_DETACH_non_wear() -> None:
    """Test the DETACH non wear function."""
    num_samples = int(1e5)
    time_list = [
        datetime(2024, 5, 2) + timedelta(milliseconds=100 * i)
        for i in range(num_samples)
    ]
    time = pl.Series("time", time_list, dtype=pl.Datetime("ns"))
    acceleration = models.Measurement(measurements=np.ones((num_samples, 3)), time=time)
    temperature = models.Measurement(
        measurements=np.random.randint(low=22, high=32, size=num_samples), time=time
    )

    expected_time_length = round(num_samples / 600)

    non_wear_array = metrics.implement_DETACH_nonwear(
        acceleration, temperature, std_criteria=0.013
    )

    assert len(non_wear_array.time) == expected_time_length
