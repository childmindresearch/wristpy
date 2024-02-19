"""Loader for the .gt3x file format."""

import pathlib

import numpy as np
import polars as pl
import pygt3x.reader
from data_model import InputData


def load(
    path: pathlib.Path,
) -> InputData:
    """Load input data from .gt3x file."""
    with pygt3x.reader.FileReader(str(path)) as reader:
        acceleration = pl.from_pandas(reader.to_pandas())
        sampling_rate = reader.info.sample_rate
        time_init = reader.to_pandas().index.to_numpy()

    time = timefix(time_init, sampling_rate)
    return InputData(
        acceleration=acceleration,
        sampling_rate=sampling_rate,
        time = time
    )

def timefix(time: np.array, sampling_rate: any) -> np.array:
    """Add ms data to timestamp data.
    
    Currently pygt3x does not save the ms data, we make the assumption that
    datapoints are saved sequentially so that each new data point is one sample 
    away from the previous.

    Args: original timestamp data from .gt3x reader
        sampling_rate: sampling rate, in Hz, from .gt3x reader

    Returns:
        raw_data: added time column with corrected ms information and 
        cast into datetime object
    """
    timestamps = time

    time_fix = [(timestamps[0] + (i / sampling_rate))*1000 
                for i in range(len(timestamps))]

    time_fix_test = np.asarray(time_fix, dtype='datetime64[ms]')

    return time_fix_test