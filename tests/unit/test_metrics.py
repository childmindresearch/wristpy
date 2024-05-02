"""Testing functions of metrics module."""
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from wristpy.core.models import Measurement
from wristpy.processing.metrics import euclidean_norm

TEST_LENGTH = 100

@pytest.mark.parametrize(
        "x,y,z, expected_enmo",
        [
            (np.zeros(TEST_LENGTH), np.zeros(TEST_LENGTH), np.zeros(TEST_LENGTH), 0),
            (np.ones(TEST_LENGTH), np.zeros(TEST_LENGTH), np.zeros(TEST_LENGTH), 0),
            (np.ones(TEST_LENGTH)*2, np.zeros(TEST_LENGTH), np.zeros(TEST_LENGTH), 1),
            (np.ones(TEST_LENGTH)*3, np.zeros(TEST_LENGTH), np.zeros(TEST_LENGTH), 2)
        ]
)
def test_euclidean_norm(
        x: np.array, y: np.array, z: np.array, expected_enmo: float
        )->None:
    """Tests the euclidean norm function."""
    dummy_date = datetime(2024, 5, 2)
    dummy_datetime = pl.Series("time", [dummy_date] * TEST_LENGTH)
    #assemble Measurement object with test array and dummy datetime
    test_acceleration = Measurement(
        measurements= np.column_stack((x, y, z)),
        time = dummy_datetime)

    enmo_results = euclidean_norm(test_acceleration)

    

    assert np.all( np.isclose(
        enmo_results.measurements, expected_enmo
        )), f"Expected {expected_enmo}"
