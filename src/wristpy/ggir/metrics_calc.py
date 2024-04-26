"""Compulation of function to calculate metrics from raw accel and temperature data."""

import numpy as np
import polars as pl

from wristpy.common.data_model import InputData, OutputData


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


def calc_epoch1_raw(output_data: OutputData) -> None:
    """Calculate mean raw acceleration signal in 5s windows, hardcoded 5s for now.

    Args:
        output_data: Output data class to grab the calibrated base metrics data.
                     The OutputData class object will be modified inplace to include
                     the new epoch1 data.
    """
    output_data.accel_epoch1 = moving_mean_fast(
        output_data.cal_acceleration, output_data.time, 5
    )


def calc_epoch1_light(input_data: InputData, output_data: OutputData) -> None:
    """Calculate mean light signal in 5s windows, hardcoded 5s for now.

    Args:
        input_data: Input data class to grab the raw light data.
        output_data: The OutputData class object will be modified inplace to include
                     the new epoch1 data.
    """
    lux_df = input_data.lux_df
    # Do not proceed with processing if there is null data
    if lux_df["lux"].is_empty() or lux_df["time"].is_empty():
        output_data.lux_upsample_epoch1 = pl.Series(
            "lux", np.zeros(len(output_data.time_epoch1))
        )
        return
    lux_mean_df = moving_mean_fast(
        pl.DataFrame(input_data.lux_df["lux"]), input_data.lux_df["time"], 5
    )

    lux_mean_df = lux_mean_df.with_columns(pl.col("time").set_sorted())

    # swap columns due to upsample format
    lux_mean_df = lux_mean_df.select(["lux_mean", "time"])
    time_match_lux, fill_df = upsample_time_match_helper(lux_mean_df, output_data)
    output_data.lux_epoch1 = pl.concat([time_match_lux, fill_df], how="vertical")[
        "lux_mean"
    ]


def calc_epoch1_battery(input_data: InputData, output_data: OutputData) -> None:
    """Calculate battery signal in 5s windows, hardcoded 5s for now.

    Args:
        input_data: Input data class to grab the raw battery data.
        output_data: The OutputData class object will be modified inplace to include
                     the new epoch1 data.
    """
    battery_df = input_data.battery_df
    # Do not proceed with processing if there is null data
    if battery_df["battery_voltage"].is_empty() or battery_df["time"].is_empty():
        output_data.battery_upsample_epoch1 = pl.Series(
            "battery_voltage", np.zeros(len(output_data.time_epoch1))
        )
        return

    battery_df = battery_df.with_columns(pl.col("time").dt.round("5s"))
    battery_df = battery_df.with_columns(pl.col("time").set_sorted())

    # swap columns due to upsample format
    battery_df = battery_df.select(["battery_voltage", "time"])
    time_match_battery, fill_df = upsample_time_match_helper(battery_df, output_data)
    output_data.battery_upsample_epoch1 = pl.concat(
        [time_match_battery, fill_df], how="vertical"
    ).select("battery_voltage")


def calc_epoch1_cap_sensor(input_data: InputData, output_data: OutputData) -> None:
    """Calculate capactive sensor in 5s windows, hardcoded 5s for now.

    Args:
        input_data: Input data class to grab the raw cap sense data.
        output_data: The OutputData class object will be modified inplace to include
                     the new epoch1 data.
    """
    cap_sense_df = input_data.capsense_df
    # Do not proceed with processing if there is null data
    if cap_sense_df["cap_sense"].is_empty() or cap_sense_df["time"].is_empty():
        output_data.capsense_upsample_epoch1 = pl.Series(
            "cap_sense", np.zeros(len(output_data.time_epoch1))
        )
        return

    cap_sense_df = cap_sense_df.with_columns(pl.col("time").set_sorted())

    def _upsample_time_match_helper(
        data_to_upsample: pl.DataFrame, output_data: OutputData
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Special helper for boolean cap sense data.

        Args:
            data_to_upsample: DataFrame with
            output_data: OutputData object containing the epoch1 time data
        """
        upsampled_data = data_to_upsample.upsample(
            time_column="time", every="5s", maintain_order=True
        ).select(pl.all().forward_fill())
        upsampled_data = upsampled_data.with_columns(pl.col("time").dt.round("5s"))

        time_epoch1_df = pl.DataFrame({"time_epoch1": output_data.time_epoch1})

        time_match_df = upsampled_data.join(
            time_epoch1_df, left_on="time", right_on="time_epoch1", how="inner"
        )

        col_name = data_to_upsample.columns[0]
        if len(time_epoch1_df) > len(time_match_df):
            diff = len(time_epoch1_df) - len(time_match_df)
            last_value = time_match_df[col_name].tail(1)
            # Append the rows of time_epoch1 corresponding to the difference
            time_epoch1_diff = time_epoch1_df.tail(diff)

            fill_df = pl.DataFrame(
                {
                    "time": time_epoch1_diff,
                    col_name: np.repeat(last_value, diff).astype(bool),
                }
            )
        else:
            fill_df = pl.DataFrame({col: [] for col in time_match_df.columns})

        return time_match_df, fill_df

    # swap columns due to upsample format
    cap_sense_df = cap_sense_df.select(["cap_sense", "time"])
    time_match_cap_sense, fill_df = _upsample_time_match_helper(
        cap_sense_df, output_data
    )
    output_data.capsense_upsample_epoch1 = pl.concat(
        [time_match_cap_sense, fill_df], how="vertical"
    )["cap_sense"]


def calc_epoch1_temp(input_data: InputData, output_data: OutputData) -> None:
    """Get the temperature data from the input data and resample to epoch1.

    Args:
        input_data: InputData object containing the temperature data.
        output_data: OutputData object to store the resampled temperature data.
    """
    temperature_df = input_data.temperature_df
    # Do not proceed with processing if there is null data
    if temperature_df["temperature"].is_empty() or temperature_df["time"].is_empty():
        output_data.temperature_upsample_epoch1 = pl.Series(
            "temperature", np.zeros(len(output_data.time_epoch1))
        )
        return

    temperature_df = temperature_df.with_columns(pl.col("time").dt.round("5s"))
    temperature_df = temperature_df.with_columns(pl.col("time").set_sorted())

    # swap columns due to upsample format
    temperature_df = temperature_df.select(["temperature", "time"])
    time_match_temperature, fill_df = upsample_time_match_helper(
        temperature_df, output_data
    )
    output_data.temperature_upsample_epoch1 = pl.concat(
        [time_match_temperature, fill_df], how="vertical"
    ).select("temperature")


def rolling_median(df: pl.DataFrame, window_size: int = 51) -> pl.DataFrame:
    """Rolling median GGIR uses for anglez calculation.

    Args:
        df: The data frame containing the acceleration data
        window_size: The desired window size, in samples, defaults to 51 samples

    Returns:
        df_rolled: dataframe with the rolling median computed of the raw signals in each
        column.
    """
    # Ensure the window size is odd, as per GGIR calculation
    if window_size % 2 == 0:
        window_size += 1

    df_lazy = df.lazy()

    def _col_rolling_median(df: pl.DataFrame, window_size: int) -> pl.DataFrame:
        """Helper function to calculate rolling median."""
        return df.select(
            [
                pl.col(column)
                .rolling_median(window_size=window_size, center=True)
                .alias(column)
                for column in df.columns
            ]
        )

    df_rolled = df_lazy.map_batches(lambda df: _col_rolling_median(df, window_size))
    return df_rolled.collect()


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
        in signal_columns. New time column that indicates the start time of
        the window that the data was averaged over.
    """
    window_size_s = str(window_size) + "s"

    full_df = pl.concat([data_df, pl.DataFrame(time_df)], how="horizontal")

    full_df = full_df.with_columns(pl.col("time").set_sorted())
    windowed_group = [
        pl.all().exclude(["time"]).drop_nans().mean().name.suffix("_mean"),
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

    This is a moving SD calculation, with non-overlapping windowindows. It will create a
    downsampled signal of the original input, includes a new time output series.

    Args:
        data_df: the data to take the standard deviation of
        time_df: the timestamps df
        window_size: The desired window size, in seconds

    Returns:
        dataframe with the moving SD of signals in each column, labeled as
        {column}_SD where column is the same as the column name
        in signal_columns. New time column that indicates the start time of
        the window that the data was averaged over.
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


def set_nonwear_flag(
    output_data: OutputData,
    window_size: int,
    window_size_long: int = 3600,
    sd_crit: float = 0.013,
    ra_crit: float = 0.05,
) -> pl.DataFrame:
    """Set non_wear_flag based on accelerometer data.

    This implements GGIR "2023" non-wear detection algorithm.
    Briefly, the algorithm, creates a sliding window of length "long_window" that steps
    forward by the short_window time.
    It checks if the acceleration data in that long window, for each axis, meets certain
    criteria thresholds to compute a non-wear value.
    And then applies that non-wear value to all the short windows that make up the long
    window. Additionally, if any of the overlapping windows have a non-wear value, that
    value is kept. Finally, there is a pass to find isolated "1s" in the non-wear value,
    and set them to 2 if surrounded by >1 values.


    Args:
        output_data: OutputData object containing accelerometer data
        window_size: Window size in seconds for grouping the data
        window_size_long: The long window size in seconds for non-wear detection
        sd_crit: Threshold criteria for standard deviation
        ra_crit: Threshold criteria for range of acceleration


    Returns:
        DataFrame with non_wear_flag indicating periods of non-wear and the time intervals
        Add the non_wear_flag to the output_data object to match the epoch1 interval
    """  # noqa: E501
    window_size_s = str(window_size) + "s"

    num_short_windows = int(window_size_long / window_size)

    def _nonwear_cond(
        df_non_wear: pl.DataFrame, sd_crit: float, ra_crit: float
    ) -> pl.Series:
        """Helper function to calculate non-wear criteria values."""
        tmp_bool = (df_non_wear["X_SD"] < sd_crit) & (df_non_wear["range_X"] < ra_crit)
        tmp_X = tmp_bool.cast(pl.Int32)

        tmp_bool = (df_non_wear["Y_SD"] < sd_crit) & (df_non_wear["range_Y"] < ra_crit)
        tmp_Y = tmp_bool.cast(pl.Int32)

        tmp_bool = (df_non_wear["Z_SD"] < sd_crit) & (df_non_wear["range_Z"] < ra_crit)
        tmp_Z = tmp_bool.cast(pl.Int32)
        NW_val = tmp_X + tmp_Y + tmp_Z

        return NW_val

    accel_time_data = pl.DataFrame(
        {
            "X": output_data.cal_acceleration["X"],
            "Y": output_data.cal_acceleration["Y"],
            "Z": output_data.cal_acceleration["Z"],
            "time": output_data.time,
        }
    )
    accel_time_data = accel_time_data.with_columns(pl.col("time").set_sorted())

    # group the acceleration data by short window length
    df_short_window = accel_time_data.group_by_dynamic(
        index_column="time", every=window_size_s
    ).agg([pl.all().exclude(["time"])])

    NW_val = np.zeros(len(df_short_window))

    for win_num in range(len(df_short_window) - num_short_windows + 1):
        # Select the rows from df_short_window to match long_window
        # GGIR uses metrics from long windows to calculate non-wear criteria

        # get the short window data that makes up the long window
        df_short_window_selected = df_short_window[
            win_num : win_num + num_short_windows
        ]

        # get the SD and range data for the long window length
        X_vals = pl.DataFrame()
        Y_vals = pl.DataFrame()
        Z_vals = pl.DataFrame()
        for curr_win in range(num_short_windows):
            X_vals = pl.concat(
                [X_vals, pl.DataFrame(df_short_window_selected["X"][curr_win])],
                how="vertical",
            )
            Y_vals = pl.concat(
                [Y_vals, pl.DataFrame(df_short_window_selected["Y"][curr_win])],
                how="vertical",
            )
            Z_vals = pl.concat(
                [Z_vals, pl.DataFrame(df_short_window_selected["Z"][curr_win])],
                how="vertical",
            )
        X_SD = X_vals.std()
        Y_SD = Y_vals.std()
        Z_SD = Z_vals.std()
        X_range = X_vals.max() - X_vals.min()
        Y_range = Y_vals.max() - Y_vals.min()
        Z_range = Z_vals.max() - Z_vals.min()

        df_long_crit = pl.DataFrame(
            {
                "X_SD": X_SD,
                "Y_SD": Y_SD,
                "Z_SD": Z_SD,
                "range_X": X_range,
                "range_Y": Y_range,
                "range_Z": Z_range,
            }
        )

        # Apply _nonwear_cond to the filtered long window data
        NW_flag_temp = _nonwear_cond(df_long_crit, sd_crit, ra_crit)

        # GGIR uses a sliding window with overlap, and compares each overlapping of the
        # short window length, if any of the overlaps have a non_wear_flag it stays.
        # Thus for each iteration we take the max value from the current window and the previous window.  # noqa: E501
        max_value = np.maximum(
            NW_val[win_num : win_num + num_short_windows],
            np.repeat(NW_flag_temp, num_short_windows),
        )
        NW_val[win_num : win_num + num_short_windows] = max_value

    # GGIR search for all cases where the non-wear value is 1, and then checks if the
    # surrounding values are > 1. If so, it sets the value to 2.
    flag_ones = np.where(NW_val == 1)[0]

    for iidx in flag_ones:
        if iidx == 0:
            continue
        if iidx == len(NW_val) - 1:
            continue
        if (NW_val[iidx - 1] > 1) and (NW_val[iidx + 1] > 1):
            NW_val[iidx] = 2

    NW_val_df = pl.DataFrame({"NW_val": NW_val})

    # Finally to create the non_wear_flag, we set the flag to 1 if the NW_val is >= 2
    # (two out of three axes are non-wear, besides some of the rewriting GGIR does)
    NW_flag_df = NW_val_df.select(
        pl.when(pl.col("NW_val") >= 2).then(1).otherwise(0).alias("non_wear_flag")
    )
    NW_flag_df = NW_flag_df.with_columns(df_short_window["time"])

    # Add the non_wear_flag to the output_data object to match the epoch_time1 interval
    # this is achieved by upsamlpling to match the temporal resolution of epoch1, and
    # then padding the last known value to non_wear_flag if there is a length mismatch
    time_match_non_wear, fill_df = upsample_time_match_helper(NW_flag_df, output_data)

    output_data.non_wear_flag_epoch1 = pl.concat(
        [time_match_non_wear, fill_df], how="vertical"
    )["non_wear_flag"]

    return NW_flag_df


def upsample_time_match_helper(
    data_to_upsample: pl.DataFrame, output_data: OutputData
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Helper function to upsample data and then to match the accel epoch1 data length.

    Args:
        data_to_upsample: DataFrame with the low frequency information that needs to be
        upsampled
        output_data: OutputData object containing the epoch1 time data
    """
    upsampled_data = data_to_upsample.upsample(
        time_column="time", every="5s", maintain_order=True
    ).select(pl.all().interpolate())

    time_epoch1_df = pl.DataFrame({"time_epoch1": output_data.time_epoch1})

    time_match_df = upsampled_data.join(
        time_epoch1_df, left_on="time", right_on="time_epoch1", how="inner"
    )

    col_name = data_to_upsample.columns[0]
    if len(time_epoch1_df) > len(time_match_df):
        diff = len(time_epoch1_df) - len(time_match_df)
        last_value = time_match_df[col_name].tail(1)
        # Append the rows of time_epoch1 corresponding to the difference
        time_epoch1_diff = time_epoch1_df.tail(diff)

        fill_df = pl.DataFrame(
            {
                "time": time_epoch1_diff,
                col_name: np.repeat(last_value, diff),
            }
        )
    else:
        fill_df = pl.DataFrame({col: [] for col in time_match_df.columns})

    return time_match_df, fill_df
