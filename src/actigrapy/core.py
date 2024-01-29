"""Main application code."""

import pathlib as pl

import actigrapy.io.exporters.csv
import actigrapy.io.loaders.gt3x


def main() -> None:
    """Dummy entry point for testing."""
    data = actigrapy.io.loaders.gt3x.load("data/test_data.gt3x")
    print("Call your main application code here")
    actigrapy.io.exporters.csv.export(data, path=pl.Path("data/test_output.csv"))


if __name__ == "__main__":
    main()
