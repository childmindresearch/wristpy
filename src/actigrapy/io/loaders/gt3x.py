"""Loader for the .gt3x file format."""

import os
import numpy as np
import pygt3x.reader

from actigrapy.common.data_model import InputData


def load(
    path: str | os.PathLike[str],
) -> InputData:
    """Load actigraphy data from .gt3x file."""
    with pygt3x.reader.FileReader(str(path)) as reader:
        df = reader.to_pandas()
    sample_rate = reader.info.sample_rate
    #get temperature data
    #temp = reader.temperature
    return df, sample_rate

def timefix(raw_data,sampling_rate):
    """currently pygt3x does not save the ms data, we make the assumption that datapoints are saved
    sequentially so that each new data point is one sample away from the previous
    assumes the raw_data df and sampling rate in Hz as input """
    timestamps = raw_data.index.to_numpy()
    time_fix = []
    for i in range(len(timestamps)):
        tmp = (timestamps[0] + ((i)/sampling_rate))*1000
        time_fix.append(tmp)
    time_fix_test = np.asarray(time_fix, dtype='datetime64[ms]')
    raw_data['time'] = time_fix_test
    return raw_data