"""Test the read_watch_data wrapper function."""

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
