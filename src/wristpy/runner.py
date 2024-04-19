"""Process gt3x data and run main plotting functionality."""

import json
import warnings
from argparse import ArgumentParser
from typing import Dict

import polars as pl

import wristpy
from wristpy.ggir import calibration, compare_dataframes, metrics_calc
from wristpy.io.loaders import gt3x

warnings.filterwarnings("ignore")


def load_config(filepath: str) -> dict:
    """Loads configurations for file paths to gt3x raw file, and ggir output file.

    Args:
        filepath: file path to config.json file.

    Returns:
        dictionary with file paths to use for finding gt3x and ggir files.
    """
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file: {filepath} not found, are you in the right directory?")
        config: Dict = {}
    return config


def update_config(filepath: str, gt3x_path: str, ggir_path: str) -> None:
    """Writes new filepaths input by users to search for gt3x file and ggir output file.

    Args:
        filepath: file path to config.json file.
        gt3x_path: new path to gt3x file.
        ggir_path: new path to ggir output file.

    Returns:
        None.
    """
    new_config = {"gt3x_raw": gt3x_path, "ggir_raw": ggir_path}
    with open(filepath, "w") as f:
        json.dump(new_config, f)


def validate_days(start: int | None, end: int | None) -> None:
    """Checks that the end day given does not occur before the start day.

    Args:
        start: the argument passed to the arg parser. The int representing where
        the dataset begins. Must start before the end day (if any)
        end: the argument passed to the arg parser. The int representing where
        the dataset ends. Must end after the start day (if any)

    Returns:
        None.
    """
    if start is not None and end is not None:
        if start > end:
            raise ValueError("the start day cannot be greater than the end day")


def run(args: str | None = None) -> None:
    """Loads, processes and plots data based on user input.

    This function loads, processes and plots data from raw gt3x files and previously
    generated GGIR outputs. The function supports a number of user-defined options for
    selecting subsets of the data by day, choosing measure types to analyze, plots to
    generate, and configuring file paths. These arguments can be accepted though the
    command line or programmatically via the function's arg parameter.

    Args:
        args: arguments fed to argparser. Default is None in the event that function
        is run via CLI.

    Returns:
        None.
    """
    parser = ArgumentParser(
        description="This is wristpy, a work in progress. At the \
                            moment you have to put in both a raw file, and a ggir \
                            output file for comparison.Please double check that \
                            default file paths are appropriately configured."
    )

    parser.add_argument(
        "gt3xfile",
        help="file name for gt3x file. Make sure \
                        file path is properly configured in config.json file. Modify \
                        path with --config or -c as needed",
        type=str,
    )

    parser.add_argument(
        "ggirfile",
        help="file name for ggir output file. Make\
                        sure that file path is properly configured in config.json file.\
                        Modify path with --config or -c as needed.",
        type=str,
    )

    parser.add_argument(
        "-s",
        "--start",
        type=int,
        help="The first day you want data\
                         as an int, e.g. 4 means the data starts on Day 4. Leave empty \
                        to start data from beginning",
    )

    parser.add_argument(
        "-e",
        "--end",
        type=int,
        help="The last day you want data\
                         as an int, e.g. 4 means the data selection ends on Day 4. \
                        Leave empty to select data through to the end.",
    )

    parser.add_argument(
        "-m",
        "--measures",
        choices=["ENMO", "anglez"],
        nargs="+",
        help="Select which measures you would liketo plot. Options \
                        include ENMO, anglez",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help=" Change file paths for \
                        gt3x raw file and ggir output file, in that order.",
        nargs=2,
    )

    parser.add_argument(
        "-p",
        "--plot",
        type=str,
        choices=["ba", "qq", "ts"],
        nargs="+",
        help='the type of plot to be displayed. Select \
                        from "ba"(bland altman plot), "qq"(quantile-quantile plot) or \
                        ts (timeseries data).',
    )

    arguments = parser.parse_args(args) if args else parser.parse_args()

    path_dict = load_config("config.json")

    # If new configuration is given, update and load the new one
    if arguments.config:
        new_gt3x_path, new_ggir_path = arguments.config
        update_config("config.json", gt3x_path=new_gt3x_path, ggir_path=new_ggir_path)
        path_dict = load_config("config.json")

    # Start day must be before End day, if both are given
    validate_days(start=arguments.start, end=arguments.end)

    gt3x_raw_path = path_dict["gt3x_raw"] + arguments.gt3xfile
    ggir_output_path = path_dict["ggir_output"] + arguments.ggirfile

    test_config = wristpy.common.data_model.Config(gt3x_raw_path, gt3x_raw_path)
    test_data = gt3x.load_fast(test_config.path_input)
    test_output = calibration.start_ggir_calibration(test_data)

    metrics_calc.calc_base_metrics(test_output)
    metrics_calc.calc_epoch1_metrics(test_output)

    ggir_data = compare_dataframes.load_ggir_output(ggir_output_path)
    ggir_data = ggir_data.rename({"timestamp": "time"})

    combo_df = compare_dataframes.compare(
        ggir_dataframe=ggir_data, wristpy_dataframe=test_output
    )

    difference_df = pl.DataFrame(
        {
            "ENMO": pl.Series(combo_df["enmo_wristpy"] - combo_df["enmo_ggir"]),
            "anglez": pl.Series(combo_df["anglez_wristpy"] - combo_df["anglez_ggir"]),
            "time": combo_df["time_epoch1"],
        }
    )

    outputdata_trimmed = pl.DataFrame(
        {
            "ENMO": combo_df["enmo_wristpy"],
            "anglez": combo_df["anglez_wristpy"],
            "time": combo_df["time_epoch1"],
        }
    )
    # If subset of dates given, select data for those dates only.
    if arguments.start or arguments.end:
        difference_df = compare_dataframes.select_days(
            df=difference_df, start_day=arguments.start, end_day=arguments.end
        )
        ggir_data = compare_dataframes.select_days(
            df=ggir_data, start_day=arguments.start, end_day=arguments.end
        )
        outputdata_trimmed = compare_dataframes.select_days(
            df=outputdata_trimmed, start_day=arguments.start, end_day=arguments.end
        )

    # for a every type of plot, for each measured selected.
    for measure in arguments.measures:
        for plot in arguments.plot:
            compare_dataframes.plot_diff(
                outputdata_trimmed=outputdata_trimmed,
                ggir_dataframe=ggir_data,
                difference_df=difference_df,
                measure=measure,
                plot=plot,
                opacity=0.5,
            )


if __name__ == "__main__":
    run()
