"""Testing ENMO computations."""

import numpy as np
import polars as pl
import pytest

from wristpy.common.data_model import OutputData
from wristpy.ggir.metrics_calc import calc_base_metrics

# set test length
test_length = int(201)


# Test cases for ENMO, when value is 0, 1, 2
@pytest.mark.parametrize(
    "x,y,z,expected_enmo",
    [
        (np.ones(test_length), np.zeros(test_length), np.zeros(test_length), 0),
        (
            np.ones(test_length) * 2,
            np.zeros(test_length),
            np.zeros(test_length),
            1,
        ),
        (np.ones(test_length) * 3, np.zeros(test_length), np.zeros(test_length), 2),
    ],
)
def test_calc_base_metrics_enmo(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, expected_enmo: float
) -> None:
    """Test calculation of ENMO.

    Args:
        x: X values.
        y: Y values.
        z: Z values.
        expected_enmo: Expected ENMO value.
    """
    output_data = OutputData()
    output_data.cal_acceleration = pl.DataFrame({"X": x, "Y": y, "Z": z})
    calc_base_metrics(output_data)
    assert np.isclose(
        output_data.enmo.to_numpy()[20], expected_enmo
    ), f"Expected ENMO to be {expected_enmo}"
