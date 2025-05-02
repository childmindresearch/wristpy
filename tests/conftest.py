"""Fixtures used by pytest."""

import pathlib
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def sample_data_gt3x() -> pathlib.Path:
    """Test data for .gt3x data file."""
    return pathlib.Path(__file__).parent / "sample_data" / "example_actigraph.gt3x"


@pytest.fixture
def sample_data_gt3x_idle_sleep_mode() -> pathlib.Path:
    """Test data for .gt3x data file."""
    return (
        pathlib.Path(__file__).parent
        / "sample_data"
        / "example_actigraph_idle_sleep_mode.gt3x"
    )


@pytest.fixture
def sample_data_bin() -> pathlib.Path:
    """Test data for .bin data file."""
    return pathlib.Path(__file__).parent / "sample_data" / "example_geneactiv.bin"


@pytest.fixture
def sample_data_txt() -> pathlib.Path:
    """Text data to test invalid file types."""
    return pathlib.Path(__file__).parent / "sample_data" / "example_text.txt"


@pytest.fixture
def actigraph_interpolation_r_version() -> pathlib.Path:
    """The actigraphy sample data processed by the mims interpolate function in R."""
    return (
        pathlib.Path(__file__).parent
        / "sample_data"
        / "actigraph_interpolation_r_version.csv"
    )


@pytest.fixture
def aggregation_r_version() -> pathlib.Path:
    """The actigraphy sample data processed by the mims aggregation function in R."""
    return pathlib.Path(__file__).parent / "sample_data" / "aggregation_r_version.csv"


@pytest.fixture
def butter_r_version() -> pathlib.Path:
    """The actigraphy sample data processed by the mims iir function in R."""
    return (
        pathlib.Path(__file__).parent
        / "sample_data"
        / "mims_butterworth_filtered_data.csv"
    )


@pytest.fixture
def mims_r_version() -> pathlib.Path:
    """The actigraphy sample data processed by the mims function in R."""
    return pathlib.Path(__file__).parent / "sample_data" / "mims_example_full_run.csv"


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
