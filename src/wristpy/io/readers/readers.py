"""Function to read accelerometer data from a file."""

import os
import pathlib
from typing import Literal, Union

import actfast
import numpy as np
import polars as pl

from wristpy.core import models


def read_watch_data(file_name: Union[pathlib.Path, str]) -> models.WatchData:
    """Read watch data from a file.

    Currently supported watch types are Actigraph .gt3x and GeneActiv .bin.

    Args:
        file_name: The filename to read the watch data from.

    Returns:
        WatchData class

    Raises: IOError if the file extension is not supported or doesn't exist.
    """
    try:
        data = actfast.read(file_name)
    except Exception as e:
        raise IOError(f"Error reading file: {e}. File type is unsupported.") from e

    measurements: dict[str, models.Measurement] = {}

    for timeseries in data["timeseries"].values():
        time = unix_epoch_time_to_polars_datetime(timeseries["datetime"])
        for sensor_name, sensor_values in timeseries.items():
            measurements[sensor_name] = models.Measurement(
                measurements=sensor_values, time=time
            )
    idle_sleep_mode_flag = False
    if os.path.splitext(file_name)[1] == ".gt3x":
        idle_sleep_mode_flag = (
            data["metadata"]["device_feature_enabled"]["sleep_mode"].lower() == "true"
        )

    return models.WatchData(
        acceleration=measurements["acceleration"],
        lux=measurements.get("light"),
        battery=measurements.get("battery_voltage"),
        capsense=measurements.get("capsense"),
        temperature=measurements.get("temperature"),
        idle_sleep_mode_flag=idle_sleep_mode_flag,
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
