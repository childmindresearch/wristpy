"""Comparing GGIR outputs to wristpy outputs for epoch 1."""

import pathlib

import numpy as np
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


def select_dates(difference_df: pl.DataFrame,
                    ggir_data: pl.DataFrame,
                    outputdata_trimmed: pl.DataFrame,
                    start:str = None,
                    end:str = None
                    )-> pl.DataFrame:
    """A function to that returns data from the dates specified if a start and/or end is given.

    Args:
        difference_df: The dataframe created by taking the difference of of wristpy's
        output and a ggir output
        ggir_data: The output of ggir
        outputdata_trimmed: The output of wristpy, trimmed to fit the length of ggir's output
        start: The optional starting point for date selection. Data is entered in the format of:
        %Y-%m-%d %H:%M:%S.
        end: The optional ending point for date selection. Data is entered in the format of:
        %Y-%m-%d %H:%M:%S.

    Returns:
        A subset of each of the three dataframes.
    """  # noqa: E501
    #create polars datetime version of start and end
    if start:
        start_datetime = pl.lit(start).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    if end:
        end_datetime = pl.lit(end).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    
    #depending on whether start and or end are provided, creates a filter that removes 
    #all dates not in the subset. Differences in the names of columns prevents 
    #consolidation, will refactor to use a mask that can be applied to all relevant
    #dataframes.
    if start and end:
        difference_df = difference_df.filter(pl.col('timestamp').is_between(start_datetime, end_datetime, closed = "both"))  # noqa: E501
        ggir_data = ggir_data.filter(pl.col("timestamp").is_between(start_datetime, end_datetime, closed = "both"))  # noqa: E501
        outputdata_trimmed = outputdata_trimmed.filter(pl.col('timestamp').is_between(start_datetime, end_datetime, closed = "both"))  # noqa: E501
    elif start:
        difference_df = difference_df.filter(pl.col('timestamp') >= start_datetime)
        ggir_data = ggir_data.filter(pl.col("timestamp") >= start_datetime)
        outputdata_trimmed = outputdata_trimmed.filter(pl.col("timestamp") >= start_datetime)  # noqa: E501
    elif end:
        difference_df = difference_df.filter(pl.col('timestamp') <= end_datetime)
        ggir_data = ggir_data.filter(pl.col("timestamp") <= end_datetime)
        outputdata_trimmed = outputdata_trimmed.filter(pl.col("timestamp") <= end_datetime)  # noqa: E501
    

    
    return difference_df, ggir_data, outputdata_trimmed



def select_days(df: pl.DataFrame, start_day: int = 0, end_day: int = None)-> pl.DataFrame:  # noqa: E501
    """Selects a subset of the dataframes, from days start:end, based on user input.

    Args:
        df: the given dataframe from which we will take data from the given data range.
        start_day: The int specifying on which day the user would like to begin taking 
        data from. If no date is given, data begins from the first day. Day 1 begins at 
        an arbitrary hour, any other start_day will begin at midnight.
        end_day: the int specifying on which day the user would like to stop taking data
        If no date is given, data is extracted through the end. The last day present in
        the dataframe ends at an arbbitrary hour, any other end_day will end just
        before midnight. 

    Returns:
        filtered_df = the subset of the input dataframe, based on the date range given.
    """
    #Make sure we're dealing with dt objects.
    df = df.with_columns(pl.col('timestamp').cast(pl.Datetime))

    min_timestamp = df['timestamp'].min()
    max_timestamp = df["timestamp"].max()
    final_day = (max_timestamp - min_timestamp).days + 1

    if start_day == 0:
        start_timestamp = min_timestamp
    else:
        start_timestamp = min_timestamp + pl.duration(days=start_day - 1)
        # Adjust start_timestamp to midnight
        start_timestamp = pl.datetime(
            start_timestamp.dt.year(),
            start_timestamp.dt.month(),
            start_timestamp.dt.day()
        )

    if end_day is not None and end_day < final_day:
        end_timestamp = min_timestamp + pl.duration(days=end_day - 1)
        # Adjust end_timestamp to just before midnight
        end_timestamp = pl.datetime(
            end_timestamp.dt.year(),
            end_timestamp.dt.month(),
            end_timestamp.dt.day()
        ) - pl.duration(microseconds=1)
    else:
        end_timestamp = max_timestamp
        if end_day > final_day:
            print(f'End Day entered is outside of data range. Last available date is {max_timestamp} which corresponds to Day {final_day}.')  # noqa: E501

    filtered_df = df.filter(pl.col('timestamp').is_between(start_timestamp, end_timestamp))  # noqa: E501
    return filtered_df
    





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
            "timestamp": pl.Series(wristpy_dataframe.time_epoch1).slice(
                idx, len(ggir_time)
            ),
            "ENMO": pl.Series(wristpy_dataframe.enmo_epoch1).slice(
                idx, len(ggir_time)
            ),
            "anglez": pl.Series(wristpy_dataframe.anglez_epoch1).slice(
                idx, len(ggir_time)
            ),
        }
    )

    difference_df = pl.DataFrame(
        {
            "ENMO": pl.Series(
                outputdata_trimmed["ENMO"] - ggir_dataframe["ENMO"]
            ),
            "anglez": pl.Series(
                outputdata_trimmed["anglez"] - ggir_dataframe["anglez"]
            ),
            "timestamp": pl.Series(outputdata_trimmed["timestamp"]),
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
        outputdata_trimmed: Dataframe with the outputData class trimmed for GGIR
        comparison
        ggir_dataframe: Dataframe with ggir data
        indices: user defined indices to plot
        opacity: For data overlay visibility

        Returns:None
    """
    fig = go.Figure()

    # Add the trimmed ENMO from outputdata_trimmed
    fig.add_trace(go.Scatter(x=difference_df["timestamp"],
                            y=outputdata_trimmed["ENMO"],
                            mode='lines',
                            line=dict(color='green', width=2),
                            name='Wristpy ENMO (Trimmed)',
                            opacity=opacity))

    # Add the ENMO from ggir_dataframe
    fig.add_trace(go.Scatter(x=difference_df["timestamp"],
                            y=ggir_dataframe["ENMO"],
                            mode='lines+markers', 
                            line=dict(color='red', dash='dash', width=2),
                            name='GGIR ENMO',
                            opacity=opacity))

    # Add the ENMO difference
    fig.add_trace(go.Scatter(x=difference_df["timestamp"],
                            y=difference_df["ENMO"],
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
    fig.add_trace(go.Scatter(x=difference_df["timestamp"],
                            y=outputdata_trimmed["anglez"],
                            mode='lines',
                            line=dict(color='green', width=2),
                            name='Wristpy Anglez (Trimmed)',
                            opacity=opacity))

    # Add the anglez from ggir_dataframe
    fig.add_trace(go.Scatter(x=difference_df["timestamp"],
                            y=ggir_dataframe["anglez"],
                            mode='lines+markers', # Change to 'lines' if you don't want markers
                            line=dict(color='red', dash='dash', width=2),
                            name='GGIR Anglez',
                            opacity=opacity
                            ))

    # Add the anglez difference
    fig.add_trace(go.Scatter(x=difference_df["timestamp"],
                            y=difference_df["anglez"],
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
    pp_x = sm.ProbPlot(output_data_trimmed['ENMO'])
    pp_y = sm.ProbPlot(ggir_data1['ENMO'])
    qqplot_2samples(pp_x, pp_y, line="r")
    plt.show()

def plot_ba(
        output_data_trimmed: pl.DataFrame,
        ggir_data1: pl.DataFrame
)->None:
    opac_dict = dict(alpha=0.5)
    f, ax = plt.subplots(1, figsize = (8,5))
    sm.graphics.mean_diff_plot(np.asarray(output_data_trimmed['ENMO']), np.asarray(ggir_data1['ENMO']), ax = ax, scatter_kwds=opac_dict)
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