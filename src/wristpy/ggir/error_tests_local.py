"""Compute error for ristpy vs GGIR."""

import os
import warnings
from typing import Tuple

import polars as pl

import wristpy
from wristpy.common.data_model import OutputData
from wristpy.ggir import calibration, compare_dataframes, metrics_calc
from wristpy.io.loaders import gt3x

warnings.filterwarnings("always")


def process_file(file_name: str, output_path: str) -> OutputData:
    """Process a gt3x file with wristpy, write csv output.

    Args:
        file_name: The path of the input file.
        output_path: The path of the output directory.

    Returns:
        None
    """
    test_config = wristpy.common.data_model.Config(file_name, output_path)
    test_data = gt3x.load_fast(test_config.path_input)
    test_output = calibration.start_ggir_calibration(test_data)
    metrics_calc.calc_base_metrics(test_output)
    metrics_calc.calc_epoch1_metrics(test_output)
    metrics_calc.calc_epoch1_raw(test_output)
    metrics_calc.set_nonwear_flag(test_output, 900)
    output_data_csv = pl.DataFrame(
        {
            "time": test_output.time_epoch1,
            "X": test_output.accel_epoch1["X_mean"],
            "Y": test_output.accel_epoch1["Y_mean"],
            "Z": test_output.accel_epoch1["Z_mean"],
            "enmo": test_output.enmo_epoch1,
            "anglez": test_output.anglez_epoch1,
            "Non-wear Flag": test_output.non_wear_flag_epoch1,
        }
    )

    output_file_path = test_config.path_output + "metrics_out.csv"

    # Check if the directory already exists
    if not os.path.exists(test_config.path_output):
        os.mkdir(test_config.path_output)
    else:
        print("Directory already exists.")

    if os.path.exists(output_file_path):
        # Generate a new filename, this only allows one copy of _new....
        base_filename = os.path.basename(output_file_path)
        filename, extension = os.path.splitext(base_filename)
        new_filename = filename + "_new" + extension

        # Write the CSV file with the new filename
        output_data_csv.write_csv(os.path.join(test_config.path_output, new_filename))
    else:
        output_data_csv.write_csv(output_file_path)

    return test_output


def compute_error(
    wristpy_data: pl.DataFrame, ggir_data: pl.DataFrame
) -> Tuple[float, float, float]:
    """Compute error between wristpy and ggir data.

    Args:
        wristpy_data: The data from wristpy.
        ggir_data: The data from ggir.

    Returns:
        A tuple containing the mean squared error for anglez, mean squared error for ENMO, and the median difference of anglez.
    """  # noqa: E501
    epoch1_data = compare_dataframes.compare_csv(
        ggir_dataframe=ggir_data, wristpy_dataframe=wristpy_data
    )

    # extend non-wear flag to smooth out edges
    epoch1_data = epoch1_data.with_columns(pl.col("time_epoch1").set_sorted())
    NW_flag_rolling_mean = epoch1_data.group_by_dynamic(
        index_column="time_epoch1", every="3h"
    ).agg(pl.col("non_wear_flag").mean())
    NW_flag_rolling_mean = NW_flag_rolling_mean["non_wear_flag"].map_elements(
        lambda x: 1 if x > 0.25 else 0, return_dtype=pl.Float64, skip_nulls=False
    )

    metrics_calc_nonwear = epoch1_data.filter(epoch1_data["non_wear_flag"] == 0)

    def _compute_mse(df: pl.DataFrame, col1: str, col2: str) -> float:
        """Helper function to compute mean squared error."""
        squared_error = (df[col1] - df[col2]) ** 2
        mse = squared_error.mean()
        return mse

    mse_anglez = _compute_mse(metrics_calc_nonwear, "anglez_wristpy", "anglez_ggir")
    mse_enmo = _compute_mse(metrics_calc_nonwear, "enmo_wristpy", "enmo_ggir")

    angz_diff = (
        metrics_calc_nonwear["anglez_wristpy"] - metrics_calc_nonwear["anglez_ggir"]
    )

    return mse_anglez, mse_enmo, angz_diff.median()
