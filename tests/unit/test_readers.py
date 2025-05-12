"""Test readers.py functions."""

import pathlib

import pytest
import actfast
from wristpy.core import models
from wristpy.io.readers import readers


def test_read_invalid_extenstion(sample_data_txt: pathlib.Path) -> None:
    """Test the read_watch_data function with an invalid file extension."""
    with pytest.raises(IOError):
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
    with pytest.raises(IOError):
        readers.read_watch_data("nonexistent_file.gt3x")


def test_extract_dynamic_range_bin(sample_data_bin: pathlib.Path) -> None:
    """Test extracting dynamic range metadata from .bin files."""
    expected_dynamic_range = (-8, 8)
    data = actfast.read(sample_data_bin)

    result = readers._extract_dynamic_range(
        metadata=data["metadata"], file_type=sample_data_bin.suffix
    )

    assert (
        result == expected_dynamic_range
    ), f"Expected dynamic range of: {expected_dynamic_range}, result was: {result}"


def test_extract_dynamic_range_gt3x(sample_data_bin: pathlib.Path) -> None:
    """Test extracting dynamic range metadata from .gt3x files."""
    expected_dynamic_range = (-8, 8)
    data = actfast.read(sample_data_bin)

    result = readers._extract_dynamic_range(
        metadata=data["metadata"], file_type=sample_data_bin.suffix
    )

    assert (
        result == expected_dynamic_range
    ), f"Expected dynamic range of: {expected_dynamic_range}, result was: {result}"
