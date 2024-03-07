"""Compulation of function to calculate metrics from raw accel and temperature data."""

import numpy as np
import polars as pl

from wristpy.common.data_model import OutputData


def calc_base_metrics(output_data: OutputData) -> OutputData:
    """Calculate the basic metrics, ENMO and angle z, from raw accelerometer data.

    Args:
        output_data: Output data class to grab the calibrated accel data.

    Returns:
        OutputData: Returns the outputData with the ENMO and anglez columns.
    """
    accel_raw = output_data.cal_acceleration

    # GGIR truncates ENMO to 0 prior to downsampling, use np, then cast to pl.DataFrame
    output_data.enmo = pl.DataFrame(
        np.maximum(np.linalg.norm(accel_raw, axis=1) - 1, 0)
    )
    output_data.enmo = output_data.enmo.rename({"column_0": "enmo"})

    # GGIR computes anglez on rolling median
    df_rolled = rolling_median(accel_raw)
    output_data.anglez = pl.DataFrame(
        np.asarray(
            np.arctan(
                df_rolled["Z"]
                / (np.sqrt(np.square(df_rolled["X"]) + np.square(df_rolled["Y"])))
            )
            / (np.pi / 180)
        )
    )
    output_data.anglez = output_data.anglez.rename({"column_0": "angle_z"})


def calc_epoch1_metrics(output_data: OutputData) -> OutputData:
    """Calculate ENMO, anglez, and time for the first epoch, hardcoded to 5s.

    Args:
        output_data: Output data class to grab the calibrated base metrics  data.

    Returns:
        OutputData: Returns the outputData with the ENMO, anglez, time columns.
    """
    enmo_tmp = moving_mean(
        output_data.enmo, output_data.time, output_data.sampling_rate, 5
    )
    anglez_tmp = moving_mean(
        output_data.anglez, output_data.time, output_data.sampling_rate, 5
    )
    output_data.enmo_epoch1 = enmo_tmp["enmo_mean"]
    output_data.time_epoch1 = enmo_tmp["window_start"]
    output_data.anglez_epoch1 = anglez_tmp["angle_z_mean"]


def down_sample(
    df: pl.DataFrame, signal_columns: list, sample_rate: int, ws: int
) -> OutputData:
    """Downsample the input signal to a desired window size, in seconds.

    This is essentially a moving mean of window length ws seconds.

    Args:
        df: The data frame containing the signals to downsample
        signal_columns: List of column names to downsample
        sample_rate: The sampling rate data was collected at
        ws: The desired window size, in seconds, of the downsampled

    Returns:
        df_ds: dataframe with the downsampled signals in each column, labeled as
        downsample_{column}, where column is the same as the column name
        in signal_columns.
    """
    ##NEEDS ERROR HANDLING FOR BAD INPUTS

    samples_per_window = int(sample_rate * ws)

    df_ds = pl.DataFrame()

    # downsample each specified column
    for column in signal_columns:
        df_ds[f"downsampled_{column}"] = (
            df[column]
            .groupby(df.index // samples_per_window)
            .mean()
            .reset_index(drop=True)
        )

    return df_ds


def rolling_median(df: pl.DataFrame, ws: int = 51) -> pl.DataFrame:
    """Rolling median GGIR uses for anglez calculation.

    Args:
        df: The data frame containing the acceleration data
        ws: The desired window size, in samples, defaults to 51 samples

    Returns:
        df_rolled: dataframe with the downsampled signals in each column, labeled as
        downsample_{column}, where column is the same as the column name
        in signal_columns.
    """
    # Ensure the window size is odd
    if ws % 2 == 0:
        ws += 1

    df_rolled = pl.DataFrame()

    # Iterate over each column and apply the rolling median
    for col in df.columns:
        rolled_col = df[col].rolling_median(window_size=ws, min_periods=1).alias(col)
        df_rolled = df_rolled.with_columns(rolled_col)

    return df_rolled


def moving_std(
    df: pl.DataFrame,
    time_df: pl.DataFrame,
    sampling_rate: int,
    ws: int,
) -> pl.DataFrame:
    """Standard deviation over specific window size.

    This is a moving STD calculation, with non-overlapping windows. It will create a
    downsampled signal of the original input, includes a new time output series.

    Args:
        df: The data frame containing the signals to downsample
        time_df: The datetime series of timestamps for the df signals
        sampling_rate: The sampling rate data was collected at
        ws: The desired window size, in seconds

    Returns:
        moving_std_df: dataframe with the moving SD computed of the raw signals in each
        column, labeled as {column}_std, where column is the same as the column name
        in signal_columns. Time column with start of window as timestamp.
    """
    # Join the time DataFrame with the original DataFrame
    full_df = pl.concat([df, pl.DataFrame(time_df)], how="horizontal")

    samples_per_window = int(ws * sampling_rate)

    # Ensure that the number of rows in the dataframe is a
    # multiple of samples_per_window
    num_rows = full_df.height
    num_full_windows = num_rows // samples_per_window
    trimmed_length = num_full_windows * samples_per_window

    # Trim the dataframe to include only full windows
    trimmed_df = full_df.head(trimmed_length)

    # Compute the moving standard deviation for each window
    # get window indexes and create window column with these indices, based on
    # counting number of samples per window computed above

    windowed_df = trimmed_df.with_row_index().with_columns(
        (pl.col("index") / samples_per_window).cast(pl.Int32).alias("window")
    )

    # For each window, get the start time, and compute SD over all columns for those samples  # noqa: E501
    moving_std_df = (
        windowed_df.group_by("window")
        .agg(
            [
                pl.col("time").first().alias("window_start"),
                pl.all().exclude(["time", "index"]).std().name.suffix("_std"),
            ]
        )
        .sort("window")
    )

    return moving_std_df


def moving_mean(
    df: pl.DataFrame,
    time_df: pl.DataFrame,
    sampling_rate: int,
    ws: int,
) -> pl.DataFrame:
    """Mean over specific window size.

    This is a moving mean calculation, with non-overlapping windows. It will create a
    downsampled signal of the original input, includes a new time output series.

    Args:
        df: The data frame containing the signals to compute moving mean
        time_df: The datetime series of timestamps for the df signals
        sampling_rate: The sampling rate data was collected at
        ws: The desired window size, in seconds

    Returns:
        dataframe with the moving mean of signals in each column, labeled as
        {column}_mean where column is the same as the column name
        in signal_columns. Column for the new time window that has start of the window.
    """
    # Join the time DataFrame with the original DataFrame
    full_df = pl.concat([df, pl.DataFrame(time_df)], how="horizontal")

    samples_per_window = int(ws * sampling_rate)

    # Ensure that the number of rows in the dataframe is a
    # multiple of samples_per_window
    num_rows = full_df.height
    num_full_windows = num_rows // samples_per_window
    trimmed_length = num_full_windows * samples_per_window

    # Trim the dataframe to include only full windows
    trimmed_df = full_df.head(trimmed_length)

    # Compute the moving standard deviation for each window
    # get window indexes and create window column with these indices, based on
    # counting number of samples per window computed above

    windowed_df = trimmed_df.with_row_index().with_columns(
        (pl.col("index") / samples_per_window).cast(pl.Int32).alias("window")
    )

    # For each window, get the start time, and compute mean over all columns for those samples  # noqa: E501
    moving_mean_df = (
        windowed_df.group_by("window")
        .agg(
            [
                pl.col("time").first().alias("window_start"),
                pl.all().exclude(["time", "index"]).mean().name.suffix("_mean"),
            ]
        )
        .sort("window")
    )

    return moving_mean_df
