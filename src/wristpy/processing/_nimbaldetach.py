"""Module for DETACH non-wear algorithm.

This module is taken, and then edited, from Adam Vert:
https://github.com/nimbal/nimbaldetach
and is licensed under the MIT License:

MIT License

Copyright (c) 2022 Adam Vert

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import polars as pl
from scipy import signal


def detach(
    acceleration_values: np.ndarray,
    temperature_values: np.ndarray,
    sampling_rate: float,
    std_thresh: float = 0.008,
    low_temperature_cutoff: float = 26,
    high_temperature_cutoff: float = 30,
    temp_dec_roc: float = -0.2,
    temp_inc_roc: float = 0.1,
    num_axes: int = 2,
) -> np.ndarray:
    """Adam Vert's implementation of the DETACH algorithm.

    Non-wear algorithm with a 5 minute minimum non-wear duration using absolute
    temperature, temperature rate of change and accelerometer standard deviation of
    3 axes to detect start and stop times of non-wear periods.

    The temperature data should be upsampled to match the accelerometer data
    sampling rate before running this function.

    Args:
        acceleration_values: 3 column numpy array with x, y, and z acceleration values.
        temperature_values: temperature values.
        sampling_rate: sampling rate for the accelerometer data, in Hz.
        std_thresh: the value which the std of an axis in the window must be below
            to trigger non-wear.
        low_temperature_cutoff: Low temperature for non-wear classification.
        high_temperature_cutoff: High temperature cut off for wear classification.
        temp_dec_roc: Decreasing temperature rate of change threshold for
            non-wear classification.
        temp_inc_roc: Increasing temperature rate of change threshold for
            wear classification.
        num_axes: The number of axes that must be below the std threshold
            to be considered nonwear.


    Returns:
        A numpy array with length of the accelerometer data marked as
            either wear (0) or non-wear (1).
    """
    window_size_1min = int(sampling_rate * 60)

    std_df = calculate_rolling_std_polars(
        acceleration=acceleration_values,
        sampling_rate=sampling_rate,
        num_axes=num_axes,
        std_thresh=std_thresh,
        window_size_1min=window_size_1min,
    )

    smoothed_temperature = lowpass_filter_signal(
        temperature_values, low_f=0.005, sample_f=sampling_rate
    )
    smoothed_temp_deg_per_min = (
        np.diff(smoothed_temperature, prepend=1) * 60 * sampling_rate
    )

    accel_temp_df = std_df.with_columns(
        [
            pl.Series("Temp_lpf", smoothed_temperature),
            pl.Series("Temp_per_min", smoothed_temp_deg_per_min),
        ]
    ).with_columns(
        [
            pl.col("Temp_lpf")
            .reverse()
            .rolling_max(window_size_1min * 5)
            .reverse()
            .alias("Max_temp_5min"),
            pl.col("Temp_lpf")
            .reverse()
            .rolling_min(window_size_1min * 5)
            .reverse()
            .alias("Min_temp_5min"),
            pl.col("Temp_per_min")
            .reverse()
            .rolling_mean(window_size_1min * 5)
            .reverse()
            .alias("Mean_temp_5min"),
        ]
    )

    candidate_nw_starts = np.flatnonzero(
        (accel_temp_df["Num_axes_fwd"] >= num_axes)
        & (accel_temp_df["Percent_above_threshold_5min"] >= 0.9)
    )

    end_crit_1 = np.flatnonzero(
        (accel_temp_df["Num_axes_bwd"] == 0)
        & (accel_temp_df["Percent_above_threshold_5min_bwd"] <= 0.50)
        & (accel_temp_df["Mean_temp_5min"] > temp_inc_roc)
    )

    end_crit_2 = np.flatnonzero(
        (accel_temp_df["Num_axes_bwd"] == 0)
        & (accel_temp_df["Percent_above_threshold_5min_bwd"] <= 0.50)
        & (accel_temp_df["Min_temp_5min"] > low_temperature_cutoff)
    )

    end_crit_combined = np.sort(np.unique(np.concatenate((end_crit_1, end_crit_2))))

    max_temp_5min = accel_temp_df["Max_temp_5min"].to_numpy()
    mean_temp_5min = accel_temp_df["Mean_temp_5min"].to_numpy()
    vert_nonwear_array = np.zeros(len(acceleration_values))
    previous_end = 0

    for start_ind in candidate_nw_starts:
        if start_ind < previous_end:
            continue

        end_ind = start_ind + window_size_1min * 5

        valid_start = (
            max_temp_5min[start_ind] < high_temperature_cutoff
            and mean_temp_5min[start_ind] < temp_dec_roc
        ) or max_temp_5min[start_ind] < low_temperature_cutoff

        if not valid_start:
            continue

        end_nw_indices = end_crit_combined[end_crit_combined > end_ind]
        bout_end_index = (
            end_nw_indices[0] if len(end_nw_indices) > 0 else accel_temp_df.height - 1
        )

        vert_nonwear_array[start_ind:bout_end_index] = 1
        previous_end = bout_end_index

    return vert_nonwear_array


def lowpass_filter_signal(
    data: np.ndarray,
    sample_f: float,
    low_f: float,
    filter_order: int = 2,
) -> np.ndarray:
    """Function that low pass filters temperature data.

    Args:
        data: 1D numpy array of data to be filtered
        sample_f: Sampling rate, in Hz.
        low_f: Low frequency cutoff for filter, in Hz.
        filter_order: order of the filter.

    Returns:
        filtered_data: 1D numpy array of filtered data
    """
    nyquist_freq = 0.5 * sample_f
    wn = low_f / nyquist_freq
    b, a = signal.butter(N=filter_order, Wn=wn, btype="lowpass")
    filtered_data = signal.filtfilt(b, a, x=data)
    return filtered_data


def calculate_rolling_std_polars(
    acceleration: np.ndarray,
    sampling_rate: float,
    window_size_1min: int,
    num_axes: int = 2,
    std_thresh: float = 0.008,
) -> pl.DataFrame:
    """Calculate rolling standard deviation for each axis of acceleration data.

    It will also calculate the number of axes below the std threshold, both forward and
    backward looking, as well as the percentage of time in the next 5 minutes that the
    number of axes below the std threshold is greater than or equal to num_axes,
    both forward and backward looking.

    Args:
        acceleration: Array containing x, y, z acceleration data.
        sampling_rate: Sampling rate of the accelerometer data in Hz.
        window_size_1min: Window size for 1 minute, in data points.
        num_axes: The number of axes that must be below the std threshold
            to be considered nonwear.
        std_thresh: the value which the std of an axis in the window must be below

    Returns:
        DataFrame with rolling standard deviations for each axis,
            both forward and backward looking.
    """
    window_size_1min = round(sampling_rate * 60)
    window_size_5min = 5 * window_size_1min

    df = pl.DataFrame(acceleration, schema=["x", "y", "z"])

    df = df.select(
        [
            pl.all().rolling_std(window_size_1min).name.suffix("_std_back"),
            pl.all()
            .reverse()
            .rolling_std(window_size_1min)
            .reverse()
            .name.suffix("_std_fwd"),
        ]
    )
    df = df.with_columns(
        [
            (
                (pl.col("x_std_fwd") < std_thresh).cast(pl.Int8)
                + (pl.col("y_std_fwd") < std_thresh).cast(pl.Int8)
                + (pl.col("z_std_fwd") < std_thresh).cast(pl.Int8)
            ).alias("Num_axes_fwd"),
            (
                (pl.col("x_std_back") < std_thresh).cast(pl.Int8)
                + (pl.col("y_std_back") < std_thresh).cast(pl.Int8)
                + (pl.col("z_std_back") < std_thresh).cast(pl.Int8)
            ).alias("Num_axes_bwd"),
        ]
    )

    return df.with_columns(
        [
            (pl.col("Num_axes_fwd") >= num_axes)
            .cast(pl.Float32)
            .reverse()
            .rolling_mean(window_size_5min)
            .reverse()
            .alias("Percent_above_threshold_5min"),
            (pl.col("Num_axes_bwd") >= num_axes)
            .cast(pl.Float32)
            .reverse()
            .rolling_mean(window_size_5min)
            .reverse()
            .alias("Percent_above_threshold_5min_bwd"),
        ]
    )
