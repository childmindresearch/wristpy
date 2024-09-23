"""Fixtures used by pytest."""

import pathlib

import pytest


@pytest.fixture
def sample_data_gt3x() -> pathlib.Path:
    """Test data for .gt3x data file."""
    return pathlib.Path(__file__).parent / "sample_data" / "example_actigraph.gt3x"


@pytest.fixture
def sample_data_bin() -> pathlib.Path:
    """Test data for .bin data file."""
    return pathlib.Path(__file__).parent / "sample_data" / "example_geneactiv.bin"


@pytest.fixture
def sample_data_txt() -> pathlib.Path:
    """#TODO: define what data. --> Test data to run."""
    return pathlib.Path(__file__).parent / "sample_data" / "example_text.txt"
