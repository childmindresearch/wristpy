"""Module for DETACH non-wear algorithm.

This module is taken from Adam Vert https://github.com/nimbal/nimbaldetach
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

from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal


def detach(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    temperature_values: np.ndarray,
    accel_freq: float = 75,
    temperature_freq: float = 0.25,
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

    Args:
        x_values: Accelerometer x values.
        y_values: Accelerometer y values.
        z_values: Accelerometer z values.
        temperature_values: temperature values.
        accel_freq: frequency of accelerometer in Hz.
        temperature_freq: frequency of temperature sensor in Hz.
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
    x_std_fwd = pd.Series(x_values)[::-1].rolling(round(accel_freq * 60)).std()[::-1]
    y_std_fwd = pd.Series(y_values)[::-1].rolling(round(accel_freq * 60)).std()[::-1]
    z_std_fwd = pd.Series(z_values)[::-1].rolling(round(accel_freq * 60)).std()[::-1]

    x_std_back = pd.Series(x_values).rolling(round(accel_freq * 60)).std()
    y_std_back = pd.Series(y_values).rolling(round(accel_freq * 60)).std()
    z_std_back = pd.Series(z_values).rolling(round(accel_freq * 60)).std()

    std_df = pd.DataFrame(
        {
            "x_std_fwd": x_std_fwd,
            "y_std_fwd": y_std_fwd,
            "z_std_fwd": z_std_fwd,
            "x_std_back": x_std_back,
            "y_std_back": y_std_back,
            "z_std_back": z_std_back,
        }
    )

    smoothed_temperature = _lowpass_filter_signal(
        temperature_values, low_f=0.005, sample_f=temperature_freq
    )
    smoothed_temp_deg_per_min = (
        np.diff(smoothed_temperature, prepend=1) * 60 * temperature_freq
    )

    std_df["Num Axes Fwd"] = np.sum(
        np.array(
            [
                std_df["x_std_fwd"] < std_thresh,
                std_df["y_std_fwd"] < std_thresh,
                std_df["z_std_fwd"] < std_thresh,
            ],
            int,
        ),
        axis=0,
    )

    std_df["Num Axes Backwards"] = np.sum(
        np.array(
            [
                std_df["x_std_back"] < std_thresh,
                std_df["y_std_back"] < std_thresh,
                std_df["z_std_back"] < std_thresh,
            ],
            int,
        ),
        axis=0,
    )

    std_df["Perc Num Axes >= num_axes for Next 5 Mins (fwd looking)"] = (
        (std_df["Num Axes Fwd"][::-1] >= num_axes)
        .rolling(int(accel_freq * 60 * 5))
        .mean()[::-1]
    )
    std_df["Perc Num Axes >= num_axes for Next 5 Mins (backward looking)"] = (
        (std_df["Num Axes Backwards"][::-1] >= num_axes)
        .rolling(int(accel_freq * 60 * 5))
        .mean()[::-1]
    )

    full_df = std_df[:: int(accel_freq / temperature_freq)]
    full_df = full_df[: len(temperature_values)]
    full_df = full_df.reset_index()
    if len(full_df) != len(smoothed_temperature):
        smoothed_temperature = smoothed_temperature[: len(full_df)]
        smoothed_temp_deg_per_min = smoothed_temp_deg_per_min[: len(full_df)]
    full_df["smoothed temperature values"] = smoothed_temperature
    full_df["Max Temp in next five mins"] = (
        full_df["smoothed temperature values"][::-1]
        .rolling(int(5 * 60 * temperature_freq))
        .max()[::-1]
    )
    full_df["Min Temp in next five mins"] = (
        full_df["smoothed temperature values"][::-1]
        .rolling(int(5 * 60 * temperature_freq))
        .min()[::-1]
    )
    full_df["smoothed temperature deg per min"] = smoothed_temp_deg_per_min
    full_df["Mean Five minute Temp Change"] = (
        full_df["smoothed temperature deg per min"][::-1]
        .rolling(int(5 * 60 * temperature_freq))
        .mean()[::-1]
    )

    candidate_nw_starts = np.where(
        (full_df["Num Axes Fwd"] >= num_axes)
        & (full_df["Perc Num Axes >= num_axes for Next 5 Mins (fwd looking)"] >= 0.9)
    )

    end_crit_1 = np.array(
        np.where(
            (full_df["Num Axes Backwards"] == 0)
            & (
                full_df["Perc Num Axes >= num_axes for Next 5 Mins (backward looking)"]
                <= 0.50
            )
            & (full_df["Mean Five minute Temp Change"] > temp_inc_roc)
        )
    )[0]

    end_crit_2 = np.array(
        np.where(
            (full_df["Num Axes Backwards"] == 0)
            & (
                full_df["Perc Num Axes >= num_axes for Next 5 Mins (backward looking)"]
                <= 0.50
            )
            & (full_df["Min Temp in next five mins"] > low_temperature_cutoff)
        )
    )[0]

    end_crit_combined = np.sort(np.unique(np.concatenate((end_crit_1, end_crit_2))))

    vert_nonwear_array = np.zeros(len(x_values))
    previous_end = 0
    for ind in candidate_nw_starts[0]:
        if ind < previous_end:
            continue
        valid_start = False
        start_ind = int(ind)
        end_ind = int(ind + temperature_freq * 60 * 5)

        if (
            full_df["Max Temp in next five mins"][start_ind] < high_temperature_cutoff
        ) & (full_df["Mean Five minute Temp Change"][start_ind] < temp_dec_roc):
            valid_start = True
        elif full_df["Max Temp in next five mins"][start_ind] < low_temperature_cutoff:
            valid_start = True

        if not valid_start:
            continue

        end_crit = end_crit_combined[end_crit_combined > end_ind]

        if len(end_crit) > 0:
            bout_end_index = end_crit[0]

        else:
            bout_end_index = full_df.last_valid_index()

        accel_start_dp = int(start_ind * accel_freq / temperature_freq)
        accel_end_dp = int(bout_end_index * accel_freq / temperature_freq)
        vert_nonwear_array[accel_start_dp:accel_end_dp] = 1

        previous_end = bout_end_index

    return vert_nonwear_array


def _lowpass_filter_signal(
    data: np.ndarray,
    sample_f: float,
    low_f: Optional[float] = None,
    filter_order: int = 2,
) -> np.ndarray:
    """Function that low pass fiters temperature data.

    Args:
        data: 1D numpy array of data to be filtered
        sample_f: Sampling rate, in Hz.
        low_f: Low frequency cutoff for filter, in Hz.
        filter_order: order of the filter.

    Returns:
        filtered_data: 1D numpy array of filtered data
    """
    nyquist_freq = 0.5 * sample_f
    low = (low_f / nyquist_freq) if low_f is not None else None
    wn = low
    b, a = signal.butter(N=filter_order, Wn=wn, btype="lowpass")
    filtered_data = signal.filtfilt(b, a, x=data)
    return filtered_data
