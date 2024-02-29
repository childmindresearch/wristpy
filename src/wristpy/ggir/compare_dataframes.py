"""Comparing GGIR outputs to wristpy outputs for epoch 1."""

import pathlib

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot_2samples

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





def plot_enmo(
    difference_df: pl.DataFrame,
    outputdata_trimmed: pl.DataFrame,
    ggir_dataframe: pl.DataFrame,
    opacity: float,
    indices: slice = slice(None),
) -> None:
    """Plot difference graphs for enmo, with user defined indices and opacity.

    Args:
        difference_df: Dataframe with time and error difference
        outputdata_trimmed: Dataframe with the outputData class trimmed for GGIR comparison
        ggir_dataframe: Dataframe with ggir data
        indices: user defined indices to plot
        opacity: For data overlay visibility

        Returns:None
    """
    fig = go.Figure()

    # Add the trimmed ENMO from outputdata_trimmed
    fig.add_trace(go.Scatter(x=difference_df["time_trimmed"],
                            y=outputdata_trimmed["trim_enmo"],
                            mode='lines',
                            line=dict(color='green', width=2),
                            name='Wristpy ENMO (Trimmed)',
                            opacity=opacity))

    # Add the ENMO from ggir_dataframe
    fig.add_trace(go.Scatter(x=difference_df["time_trimmed"],
                            y=ggir_dataframe["ENMO"],
                            mode='lines+markers', # Change to 'lines' if you don't want markers
                            line=dict(color='red', dash='dash', width=2),
                            name='GGIR ENMO',
                            opacity=opacity))

    # Add the ENMO difference
    fig.add_trace(go.Scatter(x=difference_df["time_trimmed"],
                            y=difference_df["enmo_diff"],
                            mode='lines',
                            line=dict(color='black', width=2),
                            name='ENMO Difference'))

    # Update layout if needed (e.g., titles, axes labels)
    fig.update_layout(title='ENMO Comparison',
                    xaxis_title='Time',
                    yaxis_title='ENMO Values',
                    legend_title='Legend')

    # Show the figure
    fig.show()

def plot_anglez(
    difference_df: pl.DataFrame,
    outputdata_trimmed: pl.DataFrame,
    ggir_dataframe: pl.DataFrame,
    opacity: float,
    indices: slice = slice(None),
) -> None:
    """Plot difference graphs for angelz, with user defined indices and opacity.

    Args:
        difference_df: Dataframe with time and error difference
        outputdata_trimmed: Dataframe with the outputData class trimmed for GGIR comparison
        ggir_dataframe: Dataframe with ggir data
        indices: user defined indices to plot
        opacity: For data overlay visibility

        Returns:None
    """
    fig = go.Figure()

    # Add the trimmed anglez from outputdata_trimmed
    fig.add_trace(go.Scatter(x=difference_df["time_trimmed"],
                            y=outputdata_trimmed["trim_anglez"],
                            mode='lines',
                            line=dict(color='green', width=2),
                            name='Wristpy Anglez (Trimmed)',
                            opacity=opacity))

    # Add the anglez from ggir_dataframe
    fig.add_trace(go.Scatter(x=difference_df["time_trimmed"],
                            y=ggir_dataframe["anglez"],
                            mode='lines+markers', # Change to 'lines' if you don't want markers
                            line=dict(color='red', dash='dash', width=2),
                            name='GGIR Anglez',
                            opacity=opacity
                            ))

    # Add the anglez difference
    fig.add_trace(go.Scatter(x=difference_df["time_trimmed"],
                            y=difference_df["anglez_diff"],
                            mode='lines',
                            line=dict(color='black', width=2),
                            name='Anglez Difference'))

    # Update the layout with titles and labels
    fig.update_layout(title='Anglez Comparison',
                    xaxis_title='Time',
                    yaxis_title='Anglez Values',
                    legend_title='Legend')

    # Show the figure
    fig.show()

def plot_qq(
        output_data_trimmed: pl.DataFrame,
        ggir_data1: pl.DataFrame
)->None:
    pp_x = sm.ProbPlot(output_data_trimmed['trim_enmo'])
    pp_y = sm.ProbPlot(ggir_data1['ENMO'])
    qqplot_2samples(pp_x, pp_y, line="r")
    plt.show()

def plot_ba(
        output_data_trimmed: pl.DataFrame,
        ggir_data1: pl.DataFrame
)->None:
    opac_dict = dict(alpha=0.5)
    f, ax = plt.subplots(1, figsize = (8,5))
    sm.graphics.mean_diff_plot(np.asarray(output_data_trimmed['trim_enmo']), np.asarray(ggir_data1['ENMO']), ax = ax, scatter_kwds=opac_dict)
    plt.show()
    


def plot_diff(
    difference_df: pl.DataFrame,
    outputdata_trimmed: pl.DataFrame,
    ggir_dataframe: pl.DataFrame,
    opacity: float,
    measures: str,
    indices: slice = slice(None),
) -> None:
    """Plot difference graphs, with user defined indices, opacity and measures.

    Args:
        difference_df: Dataframe with time and error difference
        outputdata_trimmed: Dataframe with the outputData class trimmed for GGIR comparison
        ggir_dataframe: Dataframe with ggir data
        indices: user defined indices to plot
        opacity: For data overlay visibility
        measures: user defined measure to plot and compare.

        Returns:None
    """
    if measures == "ENMO":
        plot_enmo(
            difference_df= difference_df,
            outputdata_trimmed= outputdata_trimmed,
            ggir_dataframe= ggir_dataframe,
            indices= indices,
            opacity = opacity)
    elif measures == "anglez":
        plot_anglez(
            difference_df= difference_df,
            outputdata_trimmed= outputdata_trimmed,
            ggir_dataframe= ggir_dataframe,
            indices= indices,
            opacity = opacity)
    elif measures == "qq":
        plot_qq(
            output_data_trimmed= outputdata_trimmed,
            ggir_data1= ggir_dataframe
        )
    elif measures == "ba":
        plot_ba(
            output_data_trimmed= outputdata_trimmed,
            ggir_data1= ggir_dataframe
        )
    else:
        print("YOU DID NOT SELECT A MEASURE!")