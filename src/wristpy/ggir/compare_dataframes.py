"""Comparing GGIR outputs to wristpy outputs for epoch 1."""

import pathlib

import numpy as np
import polars as pl
from matplotlib import pyplot as plt

from wristpy.common.data_model import OutputData


def load_ggir_output(filepath: pathlib.Path) -> pl.DataFrame:
    """Load ggir output.csv.

    Args:
        filepath: is the Path to the GGIR output .csv to load

    Returns:
        A polars data frame with the GGIR enmo, anglez, and timestamps. Timestamps have
        been sliced to remove timezone information
    """
    ggir_data = pl.read_csv(filepath)
    ggir_data = ggir_data.with_columns(pl.col("timestamp").str.slice(0, 19))

    return ggir_data


def compare(
    ggir_dataframe: pl.DataFrame, wristpy_dataframe: OutputData
) -> pl.DataFrame:
    """Compares a wristpy and ggir dataframe.

    Args:
        ggir_dataframe:
            The GGIR derived dataframe to be used to calculate difference between GGIR
            and wristpy outputs.
        wristpy_dataframe:
            The wristpy OutputData object to be used to calculate difference between
            GGIR and wristpy outputs.


    Returns:
        A difference dataframe that has the trimmed timestamps, and difference between
        GGIR output and the outputData Class enmo and anglez
    """
    ggir_time = np.asarray(ggir_dataframe["timestamp"], dtype="datetime64[ms]")
    tmp_time = np.asarray(wristpy_dataframe.time_epoch1, dtype="datetime64[ms]")

    idx = np.searchsorted(tmp_time, ggir_time[0])
    outputdata_trimmed = pl.DataFrame(
        {
            "trim_time": pl.Series(wristpy_dataframe.time_epoch1).slice(
                idx, len(ggir_time)
            ),
            "trim_enmo": pl.Series(wristpy_dataframe.enmo_epoch1).slice(
                idx, len(ggir_time)
            ),
            "trim_anglez": pl.Series(wristpy_dataframe.anglez_epoch1).slice(
                idx, len(ggir_time)
            ),
        }
    )

    difference_df = pl.DataFrame(
        {
            "enmo_diff": pl.Series(
                outputdata_trimmed["trim_enmo"] - ggir_dataframe["ENMO"]
            ),
            "anglez_diff": pl.Series(
                outputdata_trimmed["trim_anglez"] - ggir_dataframe["anglez"]
            ),
            "time_trimmed": pl.Series(outputdata_trimmed["trim_time"]),
        }
    )

    return difference_df, outputdata_trimmed


def plot_diff(
    difference_df: pl.DataFrame,
    outputdata_trimmed: pl.DataFrame,
    ggir_dataframe: pl.DataFrame,
    indices: int,
    opacity: float,
) -> None:
    """Plot difference graphs, with user defined indices and opacity.

    Args:
        difference_df: Dataframe with time and error difference
        outputdata_trimmed: Dataframe with the outputData class trimmed for GGIR comparison
        ggir_dataframe: Dataframe with ggir data
        indices: user defined indices to plot
        opacity: For data overlay visibility

        Returns:None
    """
    plt.plot(
        difference_df["time_trimmed"][indices],
        outputdata_trimmed["trim_enmo"][indices],
        alpha=opacity,
        color="b",
    )
    plt.plot(
        difference_df["time_trimmed"][indices],
        ggir_dataframe["ENMO"][indices],
        "--",
        alpha=opacity,
        color="r",
    )
    plt.plot(
        difference_df["time_trimmed"][indices],
        difference_df["enmo_diff"][indices],
    )
    plt.show()

    plt.plot(
        difference_df["time_trimmed"][indices],
        outputdata_trimmed["trim_anglez"][indices],
        alpha=opacity,
        color="b",
    )
    plt.plot(
        difference_df["time_trimmed"][indices],
        ggir_dataframe["anglez"][indices],
        "--",
        alpha=opacity,
        color="r",
    )
    plt.plot(
        difference_df["time_trimmed"][indices],
        difference_df["anglez_diff"][indices],
    )
    plt.show()
