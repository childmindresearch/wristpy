"""Compute error for ristpy vs GGIR."""

import os
import warnings
from typing import Tuple

import polars as pl

import wristpy
from wristpy.common.data_model import OutputData
from wristpy.ggir import calibration, metrics_calc
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
    metrics_calc.calc_epoch1_light(test_data, test_output)
    metrics_calc.calc_epoch1_battery(test_data, test_output)
    # metrics_calc.calc_epoch1_cap_sensor(test_data, test_output)
    output_data_csv = pl.DataFrame(
        {
            "time": test_output.time_epoch1,
            "X": test_output.accel_epoch1["X_mean"],
            "Y": test_output.accel_epoch1["Y_mean"],
            "Z": test_output.accel_epoch1["Z_mean"],
            "enmo": test_output.enmo_epoch1,
            "anglez": test_output.anglez_epoch1,
            "Non-wear Flag": test_output.non_wear_flag_epoch1,
            "light": test_output.lux_epoch1,
            "battery voltage": test_output.battery_upsample_epoch1,
            # "capacitive sensor": test_output.capsense_upsample_epoch1,
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
