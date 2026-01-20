"""Test readers.py functions."""

import pathlib
from typing import Literal, cast

import actfast
import numpy as np
import pytest
import pytest_mock

from wristpy.core import models
from wristpy.io.readers import readers


def test_read_invalid_extenstion(sample_data_txt: pathlib.Path) -> None:
    """Test the read_watch_data function with an invalid file extension."""
    with pytest.raises(ValueError, match="File type .txt is not supported."):
        readers.read_watch_data(sample_data_txt)


def test_gt3x_loader(sample_data_gt3x: pathlib.Path) -> None:
    """Test the gt3x loader."""
    watch_data = readers.read_watch_data(sample_data_gt3x)
    assert isinstance(watch_data, models.WatchData)
    assert isinstance(watch_data.acceleration, models.Measurement)
    assert isinstance(watch_data.lux, models.Measurement)
    assert isinstance(watch_data.battery, models.Measurement)
    assert isinstance(watch_data.capsense, models.Measurement)
    assert watch_data.temperature is None
    assert watch_data.idle_sleep_mode_flag is False


def test_geneactiv_bin_loader(sample_data_bin: pathlib.Path) -> None:
    """Test the geneActiv bin loader."""
    watch_data = readers.read_watch_data(sample_data_bin)
    assert isinstance(watch_data, models.WatchData)
    assert isinstance(watch_data.acceleration, models.Measurement)
    assert isinstance(watch_data.lux, models.Measurement)
    assert isinstance(watch_data.battery, models.Measurement)
    assert isinstance(watch_data.temperature, models.Measurement)
    assert watch_data.capsense is None
    assert watch_data.idle_sleep_mode_flag is False


def test_nonexistent_file() -> None:
    """Test the correct error is raised for nonexistent file."""
    with pytest.raises(IOError, match="Error reading file:"):
        readers.read_watch_data("nonexistent_file.gt3x")


def test_extract_dynamic_range_bin(sample_data_bin: pathlib.Path) -> None:
    """Test extracting dynamic range metadata from .bin files."""
    expected_dynamic_range = (-8, 8)
    data = actfast.read(sample_data_bin)
    file_type = cast(Literal[".gt3x", ".bin"], sample_data_bin.suffix)

    result = readers._extract_dynamic_range(
        metadata=data["metadata"], file_type=file_type
    )

    assert (
        result == expected_dynamic_range
    ), f"Expected dynamic range of: {expected_dynamic_range}, result was: {result}"


def test_extract_dynamic_range_gt3x(sample_data_gt3x: pathlib.Path) -> None:
    """Test extracting dynamic range metadata from .gt3x files."""
    expected_dynamic_range = (-8, 8)
    data = actfast.read(sample_data_gt3x)
    file_type = cast(Literal[".gt3x", ".bin"], sample_data_gt3x.suffix)

    result = readers._extract_dynamic_range(
        metadata=data["metadata"], file_type=file_type
    )

    assert (
        result == expected_dynamic_range
    ), f"Expected dynamic range of: {expected_dynamic_range}, result was: {result}"


def test_timezone_extraction_gt3x(sample_data_gt3x: pathlib.Path) -> None:
    """Test extracting timezone metadata from .gt3x files."""
    expected_timezone = "-05:00:00"
    watch_data = readers.read_watch_data(sample_data_gt3x)

    assert (
        watch_data.time_zone == expected_timezone
    ), f"Expected timezone of: {expected_timezone}, result was: {watch_data.time_zone}"


def test_timezone_extraction_bin(sample_data_bin: pathlib.Path) -> None:
    """Test extracting timezone metadata from .bin files."""
    expected_timezone = "-05:00"
    watch_data = readers.read_watch_data(sample_data_bin)

    assert (
        watch_data.time_zone == expected_timezone
    ), f"Expected timezone of: {expected_timezone}, result was: {watch_data.time_zone}"


@pytest.fixture(
    params=[
        (
            ".gt3x",
            {
                "timeseries": {
                    "acceleration": {
                        "datetime": np.array([1000000000, 1000000000, 2000000000]),
                        "acceleration": np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6]]),
                    },
                    "light": {
                        "datetime": np.array([1000000000, 1000000000, 2000000000]),
                        "light": np.array([10, 20, 30]),
                    },
                },
                "metadata": {
                    "device_feature_enabled": {"sleep_mode": "false"},
                    "info": {
                        "TimeZone": "-05:00:00",
                        "Acceleration Min": "-8",
                        "Acceleration Max": "8",
                    },
                },
            },
        ),
        (
            ".bin",
            {
                "timeseries": {
                    "high_frequency": {
                        "datetime": np.array([1000000000, 1000000000, 2000000000]),
                        "acceleration": np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6]]),
                    },
                    "low_frequency": {
                        "datetime": np.array([1000000000, 1000000000, 2000000000]),
                        "light": np.array([10, 20, 30]),
                    },
                },
                "metadata": {
                    "Device Capabilities": {"Accelerometer Range": "-8 to 8"},
                    "Configuration Info": {"Time Zone": "UTC-05:00"},
                },
            },
        ),
    ],
)
def duplicate_data_scenarios(request: pytest.FixtureRequest) -> tuple[str, dict]:
    """Fixture providing test data with duplicate timestamps."""
    return request.param


def test_allow_duplicates_option(
    mocker: pytest_mock.MockFixture, duplicate_data_scenarios: tuple[str, dict]
) -> None:
    """Test the allow_duplicates option in read_watch_data function."""
    file_type, mock_data = duplicate_data_scenarios
    mocker.patch("actfast.read", return_value=mock_data)

    watch_data = readers.read_watch_data(f"dummy{file_type}", allow_duplicates=True)

    assert watch_data.lux is not None
    assert len(watch_data.acceleration.time) == 2
    assert len(watch_data.lux.time) == 2
    assert np.array_equal(
        watch_data.acceleration.measurements, np.array([[1, 2, 3], [4, 5, 6]])
    )
    assert np.array_equal(watch_data.lux.measurements, np.array([10, 30]))


def test_allow_duplicates_false(
    mocker: pytest_mock.MockFixture, duplicate_data_scenarios: tuple[str, dict]
) -> None:
    """Test the allow_duplicates=False option in read_watch_data function."""
    file_type, mock_data = duplicate_data_scenarios
    mocker.patch("actfast.read", return_value=mock_data)

    with pytest.raises(ValueError, match="Time series must contain unique entries"):
        readers.read_watch_data(f"dummy{file_type}", allow_duplicates=False)
