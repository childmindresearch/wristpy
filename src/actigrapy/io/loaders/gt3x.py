"""Loader for the .gt3x file format."""

import pathlib

import numpy as np
import polars as pl
import pygt3x.reader
from wristpy.common.data_model import InputData


def load(
    path: pathlib.Path,
) -> InputData:
    """Load input data from .gt3x file.
    
    Args:
        path: path to desired .gt3x file to process

    Returns:
        df: dataframe with loaded timestamps and acceleration data from .gt3x files
        sample_rate: sampling rate in Hz
    """
    with pygt3x.reader.FileReader(str(path)) as reader:
        acceleration = pl.from_pandas(reader.to_pandas())
        sampling_rate = reader.info.sample_rate

    return InputData(
        acceleration=acceleration,
        sampling_rate=sampling_rate,
    )

def timefix(raw_data: InputData) ->InputData:
    """Add ms data to timestamp data.
    
    Currently pygt3x does not save the ms data, we make the assumption that
    datapoints are saved sequentially so that each new data point is one sample 
    away from the previous.

    Args:
        raw_data: dataframe with original timestamp data from .gt3x reader
        sampling_rate: sampling rate, in Hz, from .gt3x reader

    Returns:
        raw_data: added time column with corrected ms information and 
        cast into datetime object
    """
    timestamps = raw_data.index.to_numpy()
    time_fix = []
    for i in range(len(timestamps)):
        tmp = (timestamps[0] + ((i)/raw_data.sampling_rate))*1000
        time_fix.append(tmp)
    time_fix_test = np.asarray(time_fix, dtype='datetime64[ms]')
    raw_data['time'] = time_fix_test
    return raw_data