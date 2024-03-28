import glob
import os

import importlib
import os
import warnings

import polars as pl
from matplotlib import pyplot as plt

import wristpy
from wristpy.common import data_model
from wristpy.ggir import calibration, compare_dataframes, metrics_calc, error_tests
from wristpy.io.loaders import gt3x

warnings.filterwarnings("always")
file_path = "/Users/adam.santorelli/Downloads/raw_gt3x_data_archive/"

file_name = [
    os.path.splitext(os.path.basename(file))[0]
    for file in glob.glob(file_path + "*.gt3x")
]
dir_name = [os.path.dirname(file) for file in glob.glob(file_path + "*.gt3x")]


def main():
    for file, dir in zip(file_name, dir_name):
        input_path = dir + "/" + file + ".gt3x"
        output_path = dir + "/" + file + "_output/"
        wristpy_out = error_tests.process_file(input_path, output_path)


if __name__ == "__main__":
    main()
