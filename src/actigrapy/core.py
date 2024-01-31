"""Main application code."""

import pathlib

import actigrapy.common.data_model
import actigrapy.ggir.calibration
import actigrapy.io.exporters.csv
import actigrapy.io.loaders.gt3x


def main() -> None:
    """Dummy entry point for testing."""
    config = actigrapy.common.data_model.Config(
        path_input=pathlib.Path("data/test_data.gt3x"),
        path_output=pathlib.Path("data/test_output.csv"),
    )

    input_data = actigrapy.io.loaders.gt3x.load(config.path_input)
    output_data = actigrapy.common.data_model.OutputData()

    print("Call your main application code here")
    actigrapy.ggir.calibration.dummy_calibration(input_data, output_data)

    actigrapy.io.exporters.csv.export(output_data, path=config.path_output)


if __name__ == "__main__":
    main()
