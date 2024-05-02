"""Test the gt3x reader."""

import pathlib

import pytest

from wristpy.core.models import Measurement, WatchData
from wristpy.io.readers import readers


def test_gt3x_loader() -> None:
    """Test the gt3x loader."""
    test_file = pathlib.Path(__file__).parent / "sample_data" / "example_actigraph.gt3x"
    watch_data = readers.gt3x_loader(test_file)
    assert isinstance(watch_data, WatchData)
    assert isinstance(watch_data.acceleration, Measurement)
    assert isinstance(watch_data.lux, Measurement)
    assert watch_data.temperature is None
