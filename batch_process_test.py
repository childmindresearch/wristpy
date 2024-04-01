import glob  # noqa: D100
import os
import warnings

from wristpy.ggir import processor

warnings.filterwarnings("always")


def main() -> None:
    """Main function for batch processing."""
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
        print(f"Processing {input_path}")
        processor.process_file(input_path, output_path)


if __name__ == "__main__":
    main()
