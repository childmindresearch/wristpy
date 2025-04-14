"""Testing functions of metrics module."""

import math
import pathlib
from datetime import datetime, timedelta
from typing import List, Union

import numpy as np
import polars as pl
import pytest
from scipy import stats

from wristpy.core import models
from wristpy.io.readers import readers
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


def create_clipped_sine_data(
    cycles: int = 2, threshold: Union[float, None] = 0.95
) -> np.ndarray:
    """Fixture to create dummy sinusoidal data with clipped regions."""
    num_samples = cycles * 100
    x_values = np.linspace(0, 2 * np.pi * cycles, num_samples)
    sine_data = np.sin(x_values)

    if threshold is not None:
        sine_data = np.clip(sine_data, -threshold, threshold)

    return np.column_stack([sine_data, sine_data, sine_data])


def create_sine_data_time_stamps(
    duration_seconds: int = 2, sampling_rate: int = 100
) -> pl.Series:
    """Create timestamps for dummy sinusoidal data."""
    num_samples = duration_seconds * sampling_rate
    time_points = np.linspace(0, duration_seconds, num_samples)

    return pl.Series(
        "datetime",
        np.datetime64("2024-02-19") + (time_points * 1e9).astype("timedelta64[ns]"),
    )


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
    """Test the interpolate function for mims."""
    expected_data = pl.read_csv(actigraph_interpolation_r_version)
    expected_time = expected_data["time"].str.strptime(
        pl.Datetime("ns"), format="%Y-%m-%d %H:%M:%S%.f"
    )
    expected_ms = expected_time.dt.epoch(time_unit="ms").to_numpy()
    data = readers.read_watch_data(sample_data_gt3x)
    acceleration = data.acceleration

    interpolated_acceleration = metrics.interpolate_measure(
        acceleration=acceleration, new_frequency=100
    )
    interpolated_ms = interpolated_acceleration.time.dt.epoch(time_unit="ms").to_numpy()

    assert len(expected_time) == len(
        interpolated_acceleration.time
    ), "Timestamp series are not the same length."
    assert np.allclose(
        expected_ms, interpolated_ms, atol=10
    ), "Timestamps don't match within tolerance. "


def test_interpolate_data(
    sample_data_gt3x: pathlib.Path, actigraph_interpolation_r_version: pathlib.Path
) -> None:
    """Test the acceleration data from the interpolate."""
    expected_data = pl.read_csv(actigraph_interpolation_r_version)
    expected_acceleration = expected_data.select(["X", "Y", "Z"]).to_numpy()

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


def test_extrapolate_points() -> None:
    """Test the succesful running of extrapolate points."""
    test_data = create_clipped_sine_data()
    test_time = create_sine_data_time_stamps()
    test_measure = models.Measurement(measurements=test_data, time=test_time)
    ground_truth = create_clipped_sine_data(threshold=None)

    result = metrics.extrapolate_points(
        acceleration=test_measure, dynamic_range=(-0.95, 0.95)
    )

    assert np.allclose(result.measurements, ground_truth, rtol=3e-3)


def test_find_marker() -> None:
    """Test find markers by correctly identify likely maxed out points."""
    postive_idx = [1, 2]
    negative_idx = [7, 8]
    dummy_maxed_out_data = np.zeros(10)
    dummy_maxed_out_data[postive_idx] = 1
    dummy_maxed_out_data[negative_idx] = -1
    confidence_threshold = 0.5

    result = metrics._find_markers(
        axis=dummy_maxed_out_data, dynamic_range=(-1, 1), noise=0.03
    )

    assert np.array_equal(np.where(result > confidence_threshold)[0], postive_idx)
    assert np.array_equal(np.where(result < -confidence_threshold)[0], negative_idx)


def test_brute_force_k() -> None:
    """Test brute foce method for finding shape parameter."""
    standard_deviation = 0.09
    target_probability = 0.95
    scale = 1.0
    k_max = 0.5
    k_min = 0.01
    k_step = 0.001

    result = metrics._brute_force_k(
        standard_deviation=standard_deviation,
        target_probability=target_probability,
        scale=scale,
        k_max=k_max,
        k_min=k_min,
        k_step=k_step,
    )
    result_probability = stats.gamma.cdf(standard_deviation, a=result, scale=scale)

    assert (
        k_min < result < k_max
    ), f"Expected shape value between {k_min} and {k_max}, got: {result}"
    assert np.isclose(
        target_probability, result_probability, rtol=1e-3
    ), f"Expected target probability of: {target_probability}, got:{result_probability}"


@pytest.mark.parametrize("maxed_out_value", [-0.9, 0.9])
def test_extrapolate_neighbors(maxed_out_value: float) -> None:
    """Test extrapolate neighbors."""
    marker = np.zeros(10)
    marker[3:6] = maxed_out_value
    confidence_threshold = 0.5
    neighborhood_size = 0.1
    sampling_rate = 10

    result = metrics._extrapolate_neighbors(
        marker=marker,
        neighborhood_size=neighborhood_size,
        confidence_threshold=confidence_threshold,
        sampling_rate=sampling_rate,
    )

    assert result["start"][0] == 3
    assert result["end"][0] == 5
    assert result["left_neighborhood"][0] == 2
    assert result["right_neighborhood"][0] == 6
    assert len(result) == 1, f"Expected 1 region but got:{len(result)}"


@pytest.mark.parametrize(
    "start,end,left_neighbor, right_neighbor", [(-1, 2, -1, 3), (8, -1, 7, -1)]
)
def test_extrapolate_neighbors_out_of_range(
    monkeypatch: pytest.MonkeyPatch,
    start: int,
    end: int,
    left_neighbor: int,
    right_neighbor: int,
) -> None:
    """Test extrapolate_neighbors correctly cases where regions are out of range."""
    marker = np.zeros(10)
    neighborhood_size = 0.1
    sampling_rate = 10

    mock_edges = pl.DataFrame({"start": start, "end": end})

    monkeypatch.setattr(
        metrics, "_extrapolate_edges", lambda *args, **kwargs: mock_edges
    )

    result = metrics._extrapolate_neighbors(
        marker=marker,
        neighborhood_size=neighborhood_size,
        confidence_threshold=0.5,
        sampling_rate=sampling_rate,
    )

    assert len(result) == 1, f"Expected 1 region but got:{len(result)}"
    assert result["start"][0] == start
    assert result["end"][0] == end
    assert result["left_neighborhood"][0] == left_neighbor
    assert result["right_neighborhood"][0] == right_neighbor


def test_extrapolate_edges() -> None:
    """Test for extrapolate edges which finds edges of maxed out regions."""
    marker = np.zeros(10)
    marker[2:5] = 0.9
    marker[6:9] = -0.8
    confidence_threshold = 0.5
    sampling_rate = 10

    result = metrics._extrapolate_edges(
        marker, confidence_threshold=confidence_threshold, sampling_rate=sampling_rate
    )
    result_sorted = result.sort("start")

    assert (
        result_sorted.shape[0] == 2
    ), f"Expected 2 regions, got {result_sorted.shape[0]}."
    assert result_sorted["start"][0] == 2
    assert result_sorted["end"][0] == 4
    assert result_sorted["start"][1] == 6
    assert result_sorted["end"][1] == 8


def test_align_edges_good_case() -> None:
    """Test for align edges."""
    left = np.array([1, 8])
    right = np.array([2, 9])

    result = metrics._align_edges(
        marker_length=10,
        left=left,
        right=right,
        out_of_range_threshold=3,
        sign="hill",
    )

    assert result["start"].to_list() == left.tolist()
    assert result["end"].to_list() == right.tolist()


@pytest.mark.parametrize(
    "left, right, expected_start, expected_end",
    [
        ([1, 4, 6], [2, 5], [1, 4, 6], [2, 5, -1]),
        ([1, 4], [2, 5, 6], [-1, 1, 4], [2, 5, 6]),
    ],
)
def test_align_edges_beyond_range(
    left: List[int],
    right: List[int],
    expected_start: List[int],
    expected_end: List[int],
) -> None:
    """Test case where there's an extra index due to exceeding a boundary."""
    left_indices = np.array(left)
    right_indices = np.array(right)

    result = metrics._align_edges(
        marker_length=10,
        left=left_indices,
        right=right_indices,
        out_of_range_threshold=1,
        sign="hill",
    )

    assert result["start"].to_list() == expected_start
    assert result["end"].to_list() == expected_end


@pytest.mark.parametrize(
    "left, right, expected_start, expected_end",
    [
        ([1, 4, 6], [2, 5], [1, 4], [2, 5]),
        ([1, 4], [2, 5, 6], [1, 4], [5, 6]),
    ],
)
def test_align_edges_not_enough_samples(
    left: List[int],
    right: List[int],
    expected_start: List[int],
    expected_end: List[int],
) -> None:
    """Test case where there's an extra index, but not enough samples to keep."""
    left_indices = np.array(left)
    right_indices = np.array(right)

    result = metrics._align_edges(
        marker_length=10,
        left=left_indices,
        right=right_indices,
        out_of_range_threshold=5,
        sign="hill",
    )

    assert result["start"].to_list() == expected_start
    assert result["end"].to_list() == expected_end


@pytest.mark.parametrize("left, right,", [([1], []), ([], [1])])
def test_align_edges_empty_result(left: List[int], right: List[int]) -> None:
    """Test case where there's an extra edge, but the other list is empty."""
    left_index = np.array(left)
    right_index = np.array(right)

    result = metrics._align_edges(
        marker_length=10,
        left=left_index,
        right=right_index,
        out_of_range_threshold=10,
        sign="hill",
    )

    assert result["start"].to_list() == []
    assert result["end"].to_list() == []


def test_align_edges_mismatch() -> None:
    """Tests error when edges are mismatched by 2 or more."""
    left_indices = np.array([1, 6, 7])
    right_indices = np.array([2])

    with pytest.raises(
        ValueError, match="Mismatch in hill edges. # left: 3, # right: 1"
    ):
        metrics._align_edges(
            marker_length=10,
            left=left_indices,
            right=right_indices,
            out_of_range_threshold=1,
            sign="hill",
        )


def test_extrapolate_fit_happy_path() -> None:
    """Test the happy path where _extrapolate_fit returns a list of tuples."""
    axis = np.arange(10, dtype=float)
    time_numeric = np.arange(10, dtype=float)
    marker = np.zeros(10)
    neighbors = pl.DataFrame(
        {
            "left_neighborhood": [0],
            "start": [3],
            "end": [6],
            "right_neighborhood": [9],
        }
    )

    result = metrics._extrapolate_fit(
        axis=axis,
        time_numeric=time_numeric,
        marker=marker,
        neighbors=neighbors,
        smoothing=0.1,
        sampling_rate=10,
        neighborhood_size=0.5,
    )
    (peak_time, peak_value) = result[0]

    assert len(result) == 1, f"Expected 1 extrapolated peak, got {len(result)}."
    assert np.isclose(
        peak_time, 4.5, atol=1e-6
    ), f"Expected middle_time=4.5, got {peak_time}"
    assert np.isclose(
        peak_value, 4.5, atol=0.5
    ), f"Expected extrapolated value near 4.5, got {peak_value}"


def test_extrapolate_fit_missing_fits() -> None:
    """Test that _extrapolate_fit skips a region when a spline cannot be fitted."""
    axis = np.arange(10, dtype=float)
    time_numeric = np.arange(10, dtype=float)
    marker = np.zeros_like(axis)
    neighbors = pl.DataFrame(
        {
            "left_neighborhood": [-1],
            "start": [2],
            "end": [4],
            "right_neighborhood": [6],
        }
    )

    result = metrics._extrapolate_fit(
        axis=axis,
        time_numeric=time_numeric,
        marker=marker,
        neighbors=neighbors,
        smoothing=0.5,
        sampling_rate=10,
        neighborhood_size=0.05,
    )

    assert len(result) == 0, f"Expected no extrapolated peaks, got {len(result)}."


def test_fit_weighted_valid_region() -> None:
    """Test that fit_weighted returns spline object and prediction."""
    axis = np.arange(10, dtype=float)
    time_numeric = np.arange(10, dtype=float)
    marker = np.zeros(10)
    start = 2
    end = 5
    smoothing = 0.0
    sampling_rate = 10
    neighborhood_size = 0.5
    mid_time = (time_numeric[start] + time_numeric[end]) / 2

    spline = metrics._fit_weighted(
        axis=axis,
        time_numeric=time_numeric,
        marker=marker,
        start=start,
        end=end,
        smoothing=smoothing,
        sampling_rate=sampling_rate,
        neighborhood_size=neighborhood_size,
    )

    assert spline is not None
    predicted = spline(mid_time)
    assert np.isclose(
        predicted, 3.5, atol=0.1
    ), f"Spline predicted={predicted}, expected=3.5"


def test_fit_weighted_insufficient_data() -> None:
    """Tests that _fit_weighted returns None when spline lacks enough data."""
    axis = np.array([1.0, 2.0, 3.0])
    time_numeric = np.array([1.0, 2.0, 3.0])
    marker = np.zeros(3)
    start = 1
    end = 1

    spline = metrics._fit_weighted(
        axis=axis,
        time_numeric=time_numeric,
        marker=marker,
        start=start,
        end=end,
        smoothing=0.6,
        sampling_rate=10,
        neighborhood_size=0.05,
    )

    assert (
        spline is None
    ), f"Expected None for insufficient data (only one point). Got {spline}."


def test_fit_weighted_out_of_range() -> None:
    """Test that _fit_weighted returns None when recieving -1 indices(out of range)."""
    axis = np.arange(10, dtype=float)
    time_numeric = np.arange(10, dtype=float)
    marker = np.zeros(10)

    spline = metrics._fit_weighted(
        axis=axis,
        time_numeric=time_numeric,
        marker=marker,
        start=-1,
        end=-1,
        smoothing=0.6,
        sampling_rate=10,
        neighborhood_size=0.05,
    )

    assert spline is None, "Expected None due to invalid index values (-1)."


@pytest.mark.parametrize(
    "marker",
    [
        np.array([0, 0, 0, 0, 0.9, 0.9, 0, 0, 0, 0]),
        np.array([0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0]),
    ],
)
def test_extrapolate_interpolate(marker: np.ndarray) -> None:
    """Test _extrapolate_interpolate with and without many "confident" values."""
    axis = np.arange(10, dtype=float)
    time_numeric = np.arange(10, dtype=float)

    points = [(4.5, 4.5)]

    interpolated = metrics._extrapolate_interpolate(
        axis=axis,
        time_numeric=time_numeric,
        marker=marker,
        points=points,
        confidence_threshold=0.5,
    )

    assert np.allclose(interpolated, axis)


def test_butterworth_filter(
    sample_data_gt3x: pathlib.Path, butter_r_version: pathlib.Path
) -> None:
    """Test filter for mims. See mims_test_data_code.md for details on sample data."""
    expected_data = pl.read_csv(butter_r_version)
    expected_acceleration = expected_data.select(["IIR_X", "IIR_Y", "IIR_Z"]).to_numpy()
    test_data = readers.read_watch_data(sample_data_gt3x)
    interpolated_acceleration = metrics.interpolate_measure(
        acceleration=test_data.acceleration, new_frequency=100
    )

    filtered_data = metrics.butterworth_filter(
        acceleration=interpolated_acceleration,
        sampling_rate=100,
        cutoffs=(0.2, 5.0),
        order=4,
    )

    for axis in range(3):
        correlation = np.corrcoef(
            expected_acceleration.T[axis, :],
            filtered_data.measurements.T[axis, :],
        )
        assert np.all(
            correlation > 0.99
        ), f"Axis:{axis} did not meet the threshold, current values: {correlation}"


def test_aggregation_good(
    sample_data_gt3x: pathlib.Path, aggregation_r_version: pathlib.Path
) -> None:
    """Test MIMS aggregation. See mims_test_data_code.md for details on sample data."""
    test_data = readers.read_watch_data(sample_data_gt3x)
    test_data_interpolated = metrics.interpolate_measure(
        acceleration=test_data.acceleration, new_frequency=100
    )
    expected_results = pl.read_csv(aggregation_r_version)
    expected_acceleration = expected_results.select(
        ["AGGREGATED_X", "AGGREGATED_Y", "AGGREGATED_Z"]
    ).to_numpy()

    results = metrics.aggregate_mims(
        acceleration=test_data_interpolated, epoch=60, sampling_rate=100, rectify=True
    )

    assert np.allclose(
        expected_acceleration, results.measurements, atol=0.001
    ), f"Results did not match expectation. Results: {results.measurements}"


def test_aggregation_few_samples(
    sample_data_gt3x: pathlib.Path,
) -> None:
    """Testing scenario where there are less than the number of expected samples."""
    test_data = readers.read_watch_data(sample_data_gt3x)
    expected_acceleration = np.array([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])

    results = metrics.aggregate_mims(
        acceleration=test_data.acceleration, epoch=60, sampling_rate=100, rectify=True
    )

    assert np.all(
        expected_acceleration == results.measurements
    ), f"Results did not match expectation. Results: {results.measurements}"


def test_aggregation_rectify() -> None:
    """Test if value is set to -1 when any value is less than -150."""
    below_threshold_data = np.full((6000, 3), -200)
    dummy_date = datetime.now()
    dummy_datetime_list = [dummy_date + timedelta(seconds=i / 100) for i in range(6000)]
    below_threshold_measure = models.Measurement(
        measurements=below_threshold_data, time=pl.Series(dummy_datetime_list)
    )
    expected_acceleration = np.array([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])

    results = metrics.aggregate_mims(
        acceleration=below_threshold_measure, epoch=60, sampling_rate=100
    )

    assert np.all(
        expected_acceleration == results.measurements
    ), f"Results did not match expectation. Results: {results.measurements}"


def test_aggregation_max_value() -> None:
    """Test if value is set to -1 when max area is exceeded."""
    max_value_data = np.full((6000, 3), 100000000)
    dummy_date = datetime(2000, 1, 1)
    dummy_datetime_list = [dummy_date + timedelta(seconds=i / 100) for i in range(6000)]
    max_value_measure = models.Measurement(
        measurements=max_value_data, time=pl.Series(dummy_datetime_list)
    )
    expected_acceleration = np.array([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])

    results = metrics.aggregate_mims(
        acceleration=max_value_measure, epoch=60, sampling_rate=100
    )

    assert np.all(
        expected_acceleration == results.measurements
    ), f"Results did not match expectation. Results: {results.measurements}"
