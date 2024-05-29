"""Test readers.py functions."""

import pathlib

import pytest

from wristpy.core.models import Measurement, WatchData
from wristpy.io.readers import readers


def test_read_gt3x() -> None:
    """Test the read_watch_data function with a .gt3x file."""
    path_name = pathlib.Path(__file__).parent / "sample_data" / "example_actigraph.gt3x"
    watch_data = readers.read_watch_data(path_name)
    assert isinstance(watch_data, WatchData)
    assert isinstance(watch_data.acceleration, Measurement)


def test_read_bin() -> None:
    """Test the read_watch_data function with a .bin file."""
    path_name = pathlib.Path(__file__).parent / "sample_data" / "example_geneactiv.bin"
    watch_data = readers.read_watch_data(path_name)
    assert isinstance(watch_data, WatchData)
    assert isinstance(watch_data.acceleration, Measurement)


def test_read_invalid_extenstion() -> None:
    """Test the read_watch_data function with an invalid file extension."""
    path_name = pathlib.Path(__file__).parent / "sample_data" / "example_.txt"
    with pytest.raises(ValueError):
        readers.read_watch_data(path_name)


def test_gt3x_loader() -> None:
    """Test the gt3x loader."""
    test_file = pathlib.Path(__file__).parent / "sample_data" / "example_actigraph.gt3x"
    watch_data = readers.gt3x_loader(test_file)
    assert isinstance(watch_data, WatchData)
    assert isinstance(watch_data.acceleration, Measurement)
    assert isinstance(watch_data.lux, Measurement)
    assert watch_data.temperature is None


def test_gt3x_loader_nonexistent_file() -> None:
    """Test the correct error is raised for nonexistent file."""
    with pytest.raises(FileNotFoundError):
        readers.gt3x_loader("nonexistent_file.gt3x")


def test_gt3x_loader_wrong_extension() -> None:
    """Test the correct error is raised for wrong file extension."""
    test_file = pathlib.Path(__file__).parent / "sample_data" / "example_text.txt"
    with pytest.raises(ValueError):
        readers.gt3x_loader(test_file)


def test_geneactiv_bin_loader() -> None:
    """Test the geneActiv bin loader."""
    test_file = pathlib.Path(__file__).parent / "sample_data" / "example_geneactiv.bin"
    watch_data = readers.geneActiv_loader(test_file)
    assert isinstance(watch_data, WatchData)
    assert isinstance(watch_data.acceleration, Measurement)
    assert isinstance(watch_data.lux, Measurement)
    assert isinstance(watch_data.battery, Measurement)
    assert watch_data.capsense is None


def test_bin_loader_nonexistent_file() -> None:
    """Test the correct error is raised for nonexistent file."""
    with pytest.raises(FileNotFoundError):
        readers.geneActiv_loader("nonexistent_file.bin")


def test_bin_loader_wrong_extension() -> None:
    """Test the correct error is raised for wrong file extension."""
    test_file = pathlib.Path(__file__).parent / "sample_data" / "example_text.txt"
    with pytest.raises(ValueError):
        readers.geneActiv_loader(test_file)
