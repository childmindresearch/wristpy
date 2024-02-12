"""Compulation of function to calculate metrics from raw accel and temperature data."""

import numpy as np
import polars as pl
from numpy.linalg import norm
from wristpy.common.data_model import InputData, OutputData


def calc_base_metrics(accel_raw: InputData.accel) -> OutputData:
    """Calculate the basic metrics, ENMO and angle z, from raw accelerometer data.

    Args:
        accel_raw: The raw data containing columns 'X', 'Y', and 'Z' with
        accelerometer data.

    Returns:
        OutputData: Returns the outputData with the ENMO and anglez columns.
    """
    ENMO_calc = norm(accel_raw, axis=1) - 1
    df_rolled = rolling_median(accel_raw)
    angle_z_raw = np.asarray(
        np.arctan(
            df_rolled.Z / (np.sqrt(np.square(df_rolled.X) + np.square(df_rolled.Y)))
        )
        / (np.pi / 180)
    )

    return OutputData(ENMO=ENMO_calc, anglez=angle_z_raw)


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
    # define number of samples per window
    samples_per_window = int(sample_rate * ws)

    # initialize down_sample_pd
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

    # empty dataframe to store rollingmedian data
    df_rolled = pl.DataFrame()

    # Iterate over each column and apply the rolling median
    for col in df.columns:
        rolled_col = df[col].rolling_median(window_size=ws, min_periods=1).alias(col)
        df_rolled = df_rolled.with_columns(rolled_col)

    return df_rolled


def moving_std(
    df: pl.DataFrame,
    time_samples: InputData.time,
    sample_rate: InputData.sampling_rate,
    ws: int,
) -> pl.DataFrame:
    """Standard deviation over specific window size.

    This is a moving STD calculation, with non-overlapping windows. It will create a
    downsampled signal of the original input, includes a new time output series.

    Args:
        df: The data frame containing the signals to downsample
        time_samples: The datetime series of timestamps for the df signals
        sample_rate: The sampling rate data was collected at
        ws: The desired window size, in seconds

    Returns:
        moving_std_df: dataframe with the moving SD computed of the raw signals in each
        column, labeled as {column}_std, where column is the same as the column name
        in signal_columns.
    """
    # Create a DataFrame from the datetime array
    time_df = pl.DataFrame(time_samples)
    # to check naming columns in time_df
    time_df = time_df.rename({"column_0": "time"})

    # Join the time DataFrame with the original DataFrame
    full_df = pl.concat([df, time_df], how="horizontal")

    # Calculate the number of samples per window based on the window size in seconds

    samples_per_window = int(ws * df.sampling_rate)

    # Ensure that the number of rows in the dataframe is a
    # multiple of samples_per_window

    num_rows = full_df.height()
    num_full_windows = num_rows // samples_per_window
    trimmed_length = num_full_windows * samples_per_window

    # Trim the dataframe to include only full windows
    trimmed_df = full_df.head(trimmed_length)

    # Compute the moving standard deviation for each window

    # get window indexes and create window column with these indices, based on
    # counting number of samples per window computed above
    trimmed_df.with_row_index().with_columns(
        (pl.col("index") / samples_per_window).cast(pl.Int32).alias("window")
    )

    # For each window, get the start time, and compute SD over all columns for those samples  # noqa: E501
    moving_std_df = trimmed_df.group_by("window").agg(
        [
            pl.col("time").first().alias("window_start"),
            pl.all().exclude(["time", "index"]).std().name.suffix("_std"),
        ]
    )

    return moving_std_df


def moving_mean(
    df: pl.DataFrame,
    time_samples: InputData.time,
    sample_rate: InputData.sampling_rate,
    ws: int,
) -> pl.DataFrame:
    """Mean over specific window size.

    This is a moving mean calculation, with non-overlapping windows. It will create a
    downsampled signal of the original input, includes a new time output series.

    Args:
        df: The data frame containing the signals to compute moving mean
        time_samples: The datetime series of timestamps for the df signals
        sample_rate: The sampling rate data was collected at
        ws: The desired window size, in seconds

    Returns:
        dataframe with the moving mean of signals in each column, labeled as
        {column}_mean where column is the same as the column name
        in signal_columns.
    """
    # Create a DataFrame from the datetime array
    time_df = pl.DataFrame(time_samples)
    # to check naming columns in time_df
    time_df = time_df.rename({"column_0": "time"})

    # Join the time DataFrame with the original DataFrame
    full_df = pl.concat([df, time_df], how="horizontal")

    # Calculate the number of samples per window based on the window size in seconds

    samples_per_window = int(ws * df.sampling_rate)

    # Ensure that the number of rows in the dataframe is a
    # multiple of samples_per_window

    num_rows = full_df.height()
    num_full_windows = num_rows // samples_per_window
    trimmed_length = num_full_windows * samples_per_window

    # Trim the dataframe to include only full windows
    trimmed_df = full_df.head(trimmed_length)

    # Compute the moving standard deviation for each window

    # get window indexes and create window column with these indices, based on
    # counting number of samples per window computed above
    trimmed_df.with_row_index().with_columns(
        (pl.col("index") / samples_per_window).cast(pl.Int32).alias("window")
    )

    # For each window, get the start time, and compute mean over all columns for those samples  # noqa: E501
    moving_mean_df = trimmed_df.group_by("window").agg(
        [
            pl.col("time").first().alias("window_start"),
            pl.all().exclude(["time", "index"]).mean().name.suffix("_mean"),
        ]
    )

    return moving_mean_df
