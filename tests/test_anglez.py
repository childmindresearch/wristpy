"""Testing anglez computations."""

import numpy as np
import polars as pl
import pytest

from wristpy.common.data_model import OutputData
from wristpy.ggir.metrics_calc import calc_base_metrics

# set test length
test_length = 201


# Test cases for anglez, testing key angles, 0, 30, 45, 60, 90
@pytest.mark.parametrize(
    "x,y,z,expected_angle",
    [
        (np.ones(test_length), np.zeros(test_length), np.zeros(test_length), 0),
        (
            np.ones(test_length) * np.sqrt(3),
            np.zeros(test_length),
            np.ones(test_length),
            30,
        ),
        (np.ones(test_length), np.zeros(test_length), np.ones(test_length), 45),
        (
            np.ones(test_length),
            np.zeros(test_length),
            np.ones(test_length) * np.sqrt(3),
            60,
        ),
        (np.zeros(test_length), np.zeros(test_length), np.ones(test_length), 90),
    ],
)
def test_calc_base_metrics_anglez(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, expected_angle: float
) -> None:
    """Test calculation of angle Z.

    Args:
        x: X values.
        y: Y values.
        z: Z values.
        expected_angle: Expected angle Z value.
    """
    output_data = OutputData()
    output_data.cal_acceleration = pl.DataFrame({"X": x, "Y": y, "Z": z})
    calc_base_metrics(output_data)
    assert np.isclose(
        output_data.anglez.to_numpy()[25][0], expected_angle
    ), f"Expected angle Z to be {expected_angle}"
