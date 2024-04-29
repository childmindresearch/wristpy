"""Function to read accelermoeter data from a file."""

import pathlib

import actfast
import polars as pl
import pandas as pd

from wristpy.common.models import Measurement, WatchData


def read_watch_data(file_name: pathlib.Path | str) -> WatchData:
    """Read watch data from a file.

    This function selects the correct loader based on the file extension.
    Returns error if none of the above.

    Args:
        file_name: The filename to read the watch data from.

    Returns:
        input_data: The raw sensor data.
    """
    filename = pathlib.Path(file_name)
    if filename.suffix == ".gt3x":
        input_data = gt3x_loader(filename.as_posix())
    elif filename.suffix == ".bin":
        input_data = geneActiv_loader(filename.as_posix())
    else:
        raise ValueError(f"Unsupported file extension: {filename.suffix}")

    return input_data


def gt3x_loader(
    path: str,
) -> WatchData:
    """Load input data from .gt3x file using actfast.

        This loads the acceleration, lux, battery voltage, and capsense data.

    Args:
        path: file path to the raw data to load

    Returns:
           WatchData class
    """
    subject1 = actfast.read_actigraph_gt3x(path)

    acceleration_tmp = subject1["timeseries"]["acceleration"]["acceleration"]

    time_tmp = pl.Series(subject1["timeseries"]["acceleration"]["datetime"])
    time_actfast = pl.from_epoch(time_tmp, time_unit="ns").alias("time")

    acceleration = Measurement(measurements=acceleration_tmp, time=time_actfast)

    # light dataframe, load light data +light time
    lux_values = subject1["timeseries"]["lux"]["lux"]
    lux_time = subject1["timeseries"]["lux"]["datetime"]
    lux_datetime = pl.from_epoch(lux_time, time_unit="ns").alias("time")
    lux = Measurement(measurements=lux_values, time=lux_datetime)

    # battery voltage dataframe, load battery data + battery time
    battery_data = subject1["timeseries"]["battery_voltage"]["battery_voltage"]
    battery_time = subject1["timeseries"]["battery_voltage"]["datetime"]
    battery_datetime = pl.from_epoch(battery_time, time_unit="ns").alias("time")
    battery = Measurement(measurements=battery_data, time=battery_datetime)

    # capsense dataframe, load capsense data + capsense time
    capsense_data = subject1["timeseries"]["capsense"]["capsense"]
    capsense_time = subject1["timeseries"]["capsense"]["datetime"]
    capsense_datetime = pl.from_epoch(capsense_time, time_unit="ns").alias("time")
    cap_sense = Measurement(measurements=capsense_data, time=capsense_datetime)

    return WatchData(
        acceleration=acceleration, lux=lux, battery=battery, capsense=cap_sense
    )


def geneActiv_loader(
    path: str,
) -> WatchData:
    """Load input data from GeneActiv .bin file using actfast.

        This loads the acceleration, lux, battery voltage, and capsense data.

    Args:
        path: file path to the raw data to load

    Returns:
           InputData class
    """
    subject1 = actfast.read_geneactiv_bin(path)

    acceleration_tmp = subject1["timeseries"]["hf"]["acceleration"]

    time_tmp = pl.Series(subject1["timeseries"]["hf"]["datetime"])
    time_fast = pl.from_epoch(time_tmp, time_unit="ns").alias("time")

    acceleration = Measurement(measurements=acceleration_tmp, time=time_fast)

    # light dataframe, load light data +light time
    lux_values = subject1["timeseries"]["hf"]["light"]
    lux = Measurement(measurements=lux_values, time=time_fast)

    time_tmp_slow = pl.Series(subject1["timeseries"]["lf"]["datetime"])
    time_slow = pl.from_epoch(time_tmp_slow, time_unit="ns").alias("time")

    # battery voltage dataframe, load battery data + battery time
    battery_data = subject1["timeseries"]["lf"]["battery_voltage"]
    battery = Measurement(measurements=battery_data, time=time_slow)

    # temperature dataframe, load temperature data + temperature time
    temperature_data = subject1["timeseries"]["lf"]["temperature"]
    temperature = Measurement(measurements=temperature_data, time=time_slow)

    return WatchData(
        acceleration=acceleration,
        lux=lux,
        battery=battery,
        temperature=temperature,
    )
