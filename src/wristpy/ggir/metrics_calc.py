"""Compulation of function to calculate metrics from raw accel and temperature data."""

import numpy as np
import polars as pl

from wristpy.common.data_model import OutputData


def calc_base_metrics(output_data: OutputData) -> None:
    """Calculate the basic metrics, ENMO and angle z, from raw accelerometer data.

    Args:
        output_data: Output data class to grab the calibrated accel data. This will be
        modified inplace to include the newly calculated metrics

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


def calc_epoch1_metrics(output_data: OutputData) -> None:
    """Calculate ENMO, anglez, and time for the first epoch, hardcoded to 5s.

    Args:
        output_data: Output data class to grab the calibrated base metrics data.
                     The OutputData class object will be modified inplace to include
                     the new epoch1 data.
    """
    enmo_tmp = moving_mean_fast(output_data.enmo, output_data.time, 5)
    anglez_tmp = moving_mean_fast(output_data.anglez, output_data.time, 5)
    output_data.enmo_epoch1 = enmo_tmp["enmo_mean"]
    output_data.time_epoch1 = enmo_tmp["time"]
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


def rolling_median(df: pl.DataFrame, window_size: int = 51) -> pl.DataFrame:
    """Rolling median GGIR uses for anglez calculation.

    Args:
        df: The data frame containing the acceleration data
        window_size: The desired window size, in samples, defaults to 51 samples

    Returns:
        df_rolled: dataframe with the downsampled signals in each column, labeled as
        downsample_{column}, where column is the same as the column name
        in signal_columns.
    """
    # Ensure the window size is odd, as per GGIR calculation
    if window_size % 2 == 0:
        window_size += 1

    df_lazy = df.lazy()

    def _col_rolling_median(df: pl.DataFrame, window_size: int) -> pl.DataFrame:
        return df.select(
            [
                pl.col(column)
                .rolling_median(window_size=window_size, center=True)
                .alias(column)
                for column in df.columns
            ]
        )

    result = df_lazy.map_batches(lambda df: _col_rolling_median(df, window_size))
    return result.collect()


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


def moving_mean_fast(
    data_df: pl.DataFrame,
    time_df: pl.Series,
    window_size: int,
) -> pl.DataFrame:
    """Mean over specific window size, based on timestamps.

    This is a moving mean calculation, with non-overlapping windows. It will create a
    downsampled signal of the original input, includes a new time output series.

    Args:
        data_df: the data to take the mean of
        time_df: Direct timestamps from the raw_data
        window_size: The desired window size, in seconds

    Returns:
        dataframe with the moving mean of signals in each column, labeled as
        {column}_mean where column is the same as the column name
        in signal_columns. Column for the new time window that has start of the window.
    """
    window_size_s = str(window_size) + "s"

    full_df = pl.concat([data_df, pl.DataFrame(time_df)], how="horizontal")

    full_df = full_df.with_columns(pl.col("time").set_sorted())
    windowed_group = [
        pl.all().exclude(["time"]).mean().name.suffix("_mean"),
    ]
    df_mean = full_df.group_by_dynamic(index_column="time", every=window_size_s).agg(
        windowed_group
    )

    return df_mean


def moving_SD_fast(
    data_df: pl.DataFrame,
    time_df: pl.DataFrame,
    window_size: int,
) -> pl.DataFrame:
    """Standard deviation over specific window size, based on timestamps.

    This is a moving SD calculation, with non-overlapping windows. It will create a
    downsampled signal of the original input, includes a new time output series.

    Args:
        data_df: the data to take the mean of
        time_df: the timestamps df
        window_size: The desired window size, in seconds

    Returns:
        dataframe with the moving SD of signals in each column, labeled as
        {column}_SD where column is the same as the column name
        in signal_columns. Column for the new time window that has start of the window.
    """
    window_size_s = str(window_size) + "s"

    full_df = pl.concat([data_df, pl.DataFrame(time_df)], how="horizontal")

    full_df = full_df.with_columns(pl.col("time").set_sorted())

    windowed_group = [
        pl.all().exclude(["time"]).std().name.suffix("_SD"),
    ]

    df_SD = full_df.group_by_dynamic(index_column="time", every=window_size_s).agg(
        windowed_group
    )

    return df_SD


def set_nonwear_flag(output_data: OutputData, window_size: int) -> pl.DataFrame:
    """Set non-wear flag based on accelerometer data.

    Args:
        output_data: OutputData object containing accelerometer data
        window_size: Window size in seconds for grouping the data

    Returns:
        DataFrame with non-wear flag indicating periods of non-wear
    """
    # GGIR uses these thresholds for non-wear, sd_crit and ra_crit are criteria for STD
    # change and range of acceleration
    sd_crit = 0.013
    ra_crit = 0.05

    window_size_s = str(window_size) + "s"

    accel_time_data = pl.DataFrame(
        {
            "X": output_data.cal_acceleration["X"],
            "Y": output_data.cal_acceleration["Y"],
            "Z": output_data.cal_acceleration["Z"],
            "time_val": output_data.time,
        }
    )
    accel_time_data = accel_time_data.with_columns(pl.col("time_val").set_sorted())

    """create dataframe with the moving SD of signals in each column, labeled as
     {column}_SD and the range of each column, as these are the features GGIR uses 
    for non-wear detection"""
    df_NW = accel_time_data.group_by_dynamic(
        index_column="time_val", every=window_size_s
    ).agg(
        [
            pl.all().exclude(["time_val"]).std().name.suffix("_SD"),
            (pl.max("X") - pl.min("X")).alias("range_X"),
            (pl.max("Y") - pl.min("X")).alias("range_Y"),
            (pl.max("Z") - pl.min("X")).alias("range_Z"),
        ]
    )

    def _nonwear_cond(
        df_NW: pl.DataFrame, sd_crit: float, ra_crit: float
    ) -> pl.DataFrame:
        """Comopute non-wear condition based on GGIR criteria."""
        tmp_bool = (df_NW["X_SD"] < sd_crit) & (df_NW["range_X"] < ra_crit)
        tmp_X = tmp_bool.cast(pl.Int32)

        tmp_bool = (df_NW["Y_SD"] < sd_crit) & (df_NW["range_Y"] < ra_crit)
        tmp_Y = tmp_bool.cast(pl.Int32)

        tmp_bool = (df_NW["Z_SD"] < sd_crit) & (df_NW["range_Z"] < ra_crit)
        tmp_Z = tmp_bool.cast(pl.Int32)
        NW_val = tmp_X + tmp_Y + tmp_Z

        #  GGIR code to find ones that are isolated, and set them to 2
        flags = (NW_val == 1).arg_true()

        for iidx in flags:
            if iidx == 0:
                continue
            if iidx == len(NW_val) - 1:
                continue
            if (NW_val[iidx - 1] > 1) and (NW_val[iidx + 1] > 1):
                NW_val[iidx] = 2

        NW_flag = df_NW.select(
            pl.when(NW_val >= 2).then(1).otherwise(0).alias("Non-wear flag")
        )

        return NW_flag

    # find non-wear condition
    NW_flag = _nonwear_cond(df_NW, sd_crit, ra_crit)

    # return the nonwear flag and time columns
    NW_flag = NW_flag.with_columns(df_NW["time_val"])

    return NW_flag
