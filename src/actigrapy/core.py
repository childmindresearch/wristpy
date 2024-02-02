"""Main application code."""

import pathlib

import wristpy.common.data_model
import wristpy.ggir.calibration
import wristpy.io.exporters.csv
import wristpy.io.loaders.gt3x


def main() -> None:
    """Dummy entry point for testing."""
    config = wristpy.common.data_model.Config(
        path_input=pathlib.Path("data/test_data.gt3x"),
        path_output=pathlib.Path("data/test_output.csv"),
    )

    input_data = wristpy.io.loaders.gt3x.load(config.path_input)
    output_data = wristpy.common.data_model.OutputData()

    print("Call your main application code here")
    wristpy.ggir.calibration.dummy_calibration(input_data, output_data)

    wristpy.io.exporters.csv.export(output_data, path=config.path_output)


if __name__ == "__main__":
    main()
