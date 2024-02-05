"""Compulation of function to calculate metrics from raw accel and temperature data."""

import numpy as np
import pandas as pd
from numpy.linalg import norm

from actigrapy.common.data_model import InputData, OutputData


def calc_base_metrics(accel_raw: pd.DataFrame) -> pd.DataFrame:
    """Calculate the basic metrics, ENMO and angle z, from raw accelerometer data.

    Args:
        accel_raw: The raw data containing columns 'X', 'Y', and 'Z' with
        accelerometer data. 
    
    Returns:
        calc_data: Returns a dataframe with ENMO and anglez columns. 
    """
    ENMO_calc = norm(accel_raw, axis =1) - 1
    angle_z_raw = np.asarray(np.arctan(accel_raw.Z / (np.sqrt( np.square(accel_raw.X) + 
                                                np.square(accel_raw.Y)))) / (np.pi/180))
    calc_data = pd.DataFrame()
    calc_data['ENMO'] = ENMO_calc
    calc_data['Anglez'] = angle_z_raw
    return calc_data

def down_sample(df: pd.DataFrame, signal_columns: list, sample_rate: int, ws: int
                )-> pd.DataFrame:
    """Downsample the input signal to a desired window size, in seconds.

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
    #define number of samples per window
    samples_per_window= int(sample_rate * ws)

    #initialize down_sample_pd
    df_ds = pd.DataFrame()

    #downsample each specified column
    for column in signal_columns:
        df_ds[f'downsampled_{column}'] = (df[column].groupby
                        (df.index // samples_per_window).mean().reset_index(drop=True))
    

    return df_ds