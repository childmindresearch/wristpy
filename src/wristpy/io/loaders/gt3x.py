"""Loader for the .gt3x file format."""

import pathlib

import actfast
import numpy as np
import polars as pl
import pygt3x.reader

from wristpy.common.data_model import InputData
from wristpy.ggir import metrics_calc


def load(
    path: pathlib.Path,
) -> InputData:
    """Load input data from .gt3x file.

    Args:
        path: file path to the raw data to load

    Returns:
           InputData class
    """
    with pygt3x.reader.FileReader(str(path)) as reader:
        acceleration = pl.from_pandas(reader.to_pandas())
        sampling_rate = reader.info.sample_rate
        time_init = reader.to_pandas().index.to_numpy()

    time = timefix(time_init, sampling_rate)
    time = pl.Series(time).alias("time")
    return InputData(acceleration=acceleration, sampling_rate=sampling_rate, time=time)


def timefix(time: np.array, sampling_rate: any) -> np.array:
    """Add ms data to timestamp data.

    Currently pygt3x does not save the ms data, we make the assumption that
    datapoints are saved sequentially so that each new data point is one sample
    away from the previous.

    Args:
        time: original timestamp data from .gt3x reader
        sampling_rate: sampling rate, in Hz, from .gt3x reader

    Returns:
        raw_data: added time column with corrected ms information and
        cast into datetime object
    """
    timestamps = time

    time_fix = [
        (timestamps[0] + (i / sampling_rate)) * 1000 for i in range(len(timestamps))
    ]

    time_fix_test = np.asarray(time_fix, dtype="datetime64[ms]")

    return time_fix_test


def load_fast(
    path: pathlib.Path,
) -> InputData:
    """Load input data from .gt3x file using actfast.

    Args:
        path: file path to the raw data to load

    Returns:
           InputData class
    """
    subject1 = actfast.read_actigraph_gt3x(path)

    acceleration = pl.from_dict(
        {
            "X": subject1["timeseries"]["acceleration"]["acceleration"][:, 0],
            "Y": subject1["timeseries"]["acceleration"]["acceleration"][:, 1],
            "Z": subject1["timeseries"]["acceleration"]["acceleration"][:, 2],
        }
    )

    sampling_rate = int(subject1["metadata"]["info"]["Sample Rate"])

    time_tmp = pl.Series(subject1["timeseries"]["acceleration"]["datetime"])
    time_actfast = pl.from_epoch(time_tmp, time_unit="ns").alias("time")

    lux_values = subject1['timeseries']['lux']['lux']
    lux_time = subject1['timeseries']['lux']['datetime']
    lux_datetime = pl.from_epoch(lux_time, time_unit= 'ns').alias('time')
    lux_df = pl.DataFrame(
        {
            'lux': lux_values,
            'time': lux_datetime
        }
    )
    #temporal resolution, hardcoded to 5 seconds for now 
    lux_df_mean = metrics_calc.moving_mean_fast(pl.DataFrame(lux_df['lux']), lux_df['time'], 5)

    #battery voltage dataframe
    battery_data = subject1['timeseries']['battery_voltage']['battery_voltage']
    battery_time = subject1['timeseries']['battery_voltage']['datetime']
    battery_datetime = pl.from_epoch(battery_time, time_unit= 'ns').alias('time')
    battery_df = pl.DataFrame(
        {
            'battery_voltage': battery_data,
            "time": battery_datetime
        }
    )
    #set flag to sorted for upsampling function
    battery_df = battery_df.set_sorted(column='time')
    #upsample so we have a temporal resolution of 5s (previously 1 minute)
    battery_df_upsample = battery_df.upsample(time_column= 'time', every = '5s', maintain_order= True).select(pl.all().forward_fill())


    #capsense dataframe 
    capsense_data = subject1['timeseries']['capsense']['capsense']
    capsense_time = subject1['timeseries']['capsense']['datetime']
    capsense_datetime = pl.from_epoch(capsense_time, time_unit = 'ns').alias('time')
    capsense_df = pl.DataFrame(
        {
            'capsense' : capsense_data,
            'time' : capsense_datetime
        }
    )

    capsense_df = capsense_df.set_sorted(column='time')
    capsense_df_upsample = capsense_df.upsample(time_column= 'time', every = '5s', maintain_order= True).select(pl.all().forward_fill())


    return InputData(
        acceleration=acceleration, 
        sampling_rate=sampling_rate, 
        time=time_actfast, 
        lux_df= lux_df,
        lux_df_mean= lux_df_mean,
        battery_df = battery_df,
        battery_df_upsample  = battery_df_upsample,
        capsense_df = capsense_df,
        capsense_df_upsample = capsense_df_upsample
    )