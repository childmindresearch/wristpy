"""Main application code."""

import pathlib as pl

import pandas as pd

import actigrapy.ggir.calibration
import actigrapy.io.exporters.csv
import actigrapy.io.loaders.gt3x


def main() -> None:
    """Dummy entry point for testing."""
    input_data = actigrapy.io.loaders.gt3x.load("data/test_data.gt3x")
    output_data = pd.DataFrame()

    print("Call your main application code here")
    actigrapy.ggir.calibration.dummy_calibration(input_data, output_data)

    actigrapy.io.exporters.csv.export(output_data, path=pl.Path("data/test_output.csv"))


if __name__ == "__main__":
    main()
