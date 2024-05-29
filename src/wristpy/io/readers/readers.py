"""Function to read accelermoeter data from a file."""

import pathlib
from typing import Literal

import actfast
import numpy as np
import polars as pl

from wristpy.core import models


def read_watch_data(file_name: pathlib.Path | str) -> models.WatchData:
    """Read watch data from a file.

    This function selects the correct loader based on the file extension.
    Returns error if none of the above.

    Args:
        file_name: The filename to read the watch data from.

    Returns:
        WatchData class

    Raises: ValueError if the file extension is not supported.
    """
    filename = pathlib.Path(file_name)
    if filename.suffix == ".gt3x":
        return gt3x_loader(filename)
    elif filename.suffix == ".bin":
        return geneActiv_loader(filename)
    raise ValueError(f"Unsupported file extension: {filename.suffix}")


def gt3x_loader(
    path: pathlib.Path | str,
) -> models.WatchData:
    """Load input data from .gt3x file using actfast.

        This loads the acceleration, lux, battery voltage, and capsense data.

    Args:
        path: file path to the raw data to load

    Returns:
           WatchData class

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file extension is not .gt3x.
    """
    file_path = pathlib.Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    if file_path.suffix != ".gt3x":
        raise ValueError(f"The file {file_path} is not a .gt3x file.")

    subject1 = actfast.read(file_path)

    acceleration_tmp = subject1["timeseries"]["acceleration"]["acceleration"]
    time_actfast = unix_epoch_time_to_polars_datetime(
        subject1["timeseries"]["acceleration"]["datetime"]
    )

    acceleration = models.Measurement(measurements=acceleration_tmp, time=time_actfast)

    # light dataframe, load light data +light time
    lux_values = subject1["timeseries"]["light"]["light"]
    lux_datetime = unix_epoch_time_to_polars_datetime(
        subject1["timeseries"]["light"]["datetime"]
    )

    lux = models.Measurement(measurements=lux_values, time=lux_datetime)

    battery_data = subject1["timeseries"]["battery_voltage"]["battery_voltage"]
    battery_datetime = unix_epoch_time_to_polars_datetime(
        subject1["timeseries"]["battery_voltage"]["datetime"]
    )

    battery = models.Measurement(measurements=battery_data, time=battery_datetime)

    # capsense (skin/wear detection) dataframe, load capsense data + capsense time
    capsense_data = subject1["timeseries"]["capsense"]["capsense"]
    capsense_datetime = unix_epoch_time_to_polars_datetime(
        subject1["timeseries"]["capsense"]["datetime"]
    )

    cap_sense = models.Measurement(measurements=capsense_data, time=capsense_datetime)

    return models.WatchData(
        acceleration=acceleration, lux=lux, battery=battery, capsense=cap_sense
    )


def geneActiv_loader(
    path: pathlib.Path | str,
) -> models.WatchData:
    """Load input data from GeneActiv .bin file using actfast.

        This loads the acceleration, lux, battery voltage, and temperature data.
        geneActiv bin file has two different time scales for different sensors, we load
        them here as fast (higher sampling rate) and slow (lower sampling rate).

    Args:
        path: file path to the raw data to load

    Returns:
           WatchData class

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file extension is not .bin.
    """
    file_path = pathlib.Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    if file_path.suffix != ".bin":
        raise ValueError(f"The file {file_path} is not a .bin file.")

    subject1 = actfast.read(file_path)

    time_fast = unix_epoch_time_to_polars_datetime(
        subject1["timeseries"]["high_frequency"]["datetime"]
    )
    time_slow = unix_epoch_time_to_polars_datetime(
        subject1["timeseries"]["low_frequency"]["datetime"]
    )

    acceleration_tmp = subject1["timeseries"]["high_frequency"]["acceleration"]
    acceleration = models.Measurement(measurements=acceleration_tmp, time=time_fast)

    # light dataframe, load light data +light time
    lux_values = subject1["timeseries"]["high_frequency"]["light"]
    lux = models.Measurement(measurements=lux_values, time=time_fast)

    battery_data = subject1["timeseries"]["low_frequency"]["battery_voltage"]
    battery = models.Measurement(measurements=battery_data, time=time_slow)

    temperature_data = subject1["timeseries"]["low_frequency"]["temperature"]
    temperature = models.Measurement(measurements=temperature_data, time=time_slow)

    return models.WatchData(
        acceleration=acceleration,
        lux=lux,
        battery=battery,
        temperature=temperature,
    )


def unix_epoch_time_to_polars_datetime(
    time: np.ndarray, units: Literal["ns", "us", "ms", "s", "d"] = "ns"
) -> pl.Series:
    """Convert unix epoch time to polars Series of datetime.

    Args:
        time: The unix epoch timestamps to convert.
        units: The units to convert the time to ('s', 'ms', 'us', or 'ns'). Default
        value is 'ns'.
    """
    time_series = pl.Series(time)
    return pl.from_epoch(time_series, time_unit=units).alias("time")
