import glob
import os
import warnings

import polars as pl

from wristpy.ggir import processor

warnings.filterwarnings("always")


def main():
    file_path = "/app/data/"
    output_dir = "/app/output/"
    file_name = sorted(
        [
            os.path.splitext(os.path.basename(file))[0]
            for file in glob.glob(file_path + "*.gt3x")
        ]
    )
    for file in file_name:
        input_path = file_path + file + ".gt3x"
        output_path = output_dir + file + "_output/"
        processor.process_file(input_path, output_path)


if __name__ == "__main__":
    main()
