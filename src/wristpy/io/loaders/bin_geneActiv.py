"""Loader for the .bin file format."""

import pathlib

import actfast
import numpy as np
import polars as pl

from wristpy.common.data_model import InputData


def load_fast_bin(
    path: pathlib.Path,
) -> InputData:
    """Load input data from GeneActiv .bin file using actfast.

        This loads the acceleration, lux, battery voltage, and capsense data.

    Args:
        path: file path to the raw data to load

    Returns:
           InputData class
    """
    subject1 = actfast.read_geneactiv_bin(path)

    acceleration = pl.from_dict(
        {
            "X": subject1["timeseries"]["hf"]["acceleration"][:, 0],
            "Y": subject1["timeseries"]["hf"]["acceleration"][:, 1],
            "Z": subject1["timeseries"]["hf"]["acceleration"][:, 2],
        }
    )

    sampling_rate = int(
        subject1["metadata"]["configuration_measurement_frequency"][0:3]
    )

    time_tmp = pl.Series(subject1["timeseries"]["hf"]["datetime"])
    time_actfast = pl.from_epoch(time_tmp, time_unit="ns").alias("time")

    # light dataframe, load light data +light time
    lux_values = subject1["timeseries"]["hf"]["light"]
    lux_df = pl.DataFrame({"lux": lux_values, "time": time_actfast})

    time_tmp_slow = pl.Series(subject1["timeseries"]["lf"]["datetime"])
    time_slow = pl.from_epoch(time_tmp_slow, time_unit="ns").alias("time")

    # battery voltage dataframe, load battery data + battery time
    battery_data = subject1["timeseries"]["lf"]["battery_voltage"]
    battery_df = pl.DataFrame({"battery_voltage": battery_data, "time": time_slow})

    # temperature dataframe, load temperature data + temperature time
    temperature_data = subject1["timeseries"]["lf"]["temperature"]
    temperature_df = pl.DataFrame({"temperature": temperature_data, "time": time_slow})

    return InputData(
        acceleration=acceleration,
        sampling_rate=sampling_rate,
        time=time_actfast,
        lux_df=lux_df,
        battery_df=battery_df,
        temperature_df=temperature_df,
    )
