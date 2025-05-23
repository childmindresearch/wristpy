"""Test the writers module."""

import datetime
import logging
import pathlib

import numpy as np
import polars as pl
import pytest

from wristpy.core import models
from wristpy.io.writers import writers


@pytest.fixture
def dummy_results() -> writers.OrchestratorResults:
    """Makes a results object for the purpose of testing."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_measure = models.Measurement(
        measurements=np.random.rand(100),
        time=pl.Series(
            [dummy_date + datetime.timedelta(seconds=i) for i in range(100)]
        ),
    )
    dummy_results = writers.OrchestratorResults(
        physical_activity_metric=dummy_measure,
        anglez=dummy_measure,
        physical_activity_levels=dummy_measure,
        nonwear_status=dummy_measure,
        sleep_status=dummy_measure,
    )

    return dummy_results


def test_empty_params(
    dummy_results: writers.OrchestratorResults, caplog: pytest.LogCaptureFixture
) -> None:
    """Test empty params raises logger warning."""
    caplog.set_level(logging.WARNING)
    dummy_results.processing_params = {}

    dummy_results.save_config_as_json(pathlib.Path("test_output.csv"))

    assert "No processing parameters to save as JSON" in caplog.text
