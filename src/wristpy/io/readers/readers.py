"""Function to read accelerometer data from a file."""

import pathlib
from typing import Literal

import actfast
import numpy as np
import polars as pl

from wristpy.core import models


def read_watch_data(file_name: pathlib.Path | str) -> models.WatchData:
    """Read watch data from a file.

    Args:
        file_name: The filename to read the watch data from.

    Returns:
        WatchData class

    Raises: ValueError if the file extension is not supported.
    """
    try:
        data = actfast.read(file_name)
    except Exception as e:
        raise ValueError(f"Error reading file: {e}") from e

    measurements: dict[str, models.Measurement] = {}

    for timeseries in data["timeseries"].values():
        time = unix_epoch_time_to_polars_datetime(timeseries["datetime"])
        for sensor in [
            "acceleration",
            "light",
            "battery_voltage",
            "capsense",
            "temperature",
        ]:
            values = timeseries.get(sensor)
            if values is not None:
                measurements[sensor] = models.Measurement(
                    measurements=values, time=time
                )

    return models.WatchData(
        acceleration=measurements["acceleration"],
        lux=measurements.get("light"),
        battery=measurements.get("battery_voltage"),
        capsense=measurements.get("capsense"),
        temperature=measurements.get("temperature"),
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
