"""Function to read accelerometer data from a file."""

import pathlib
from typing import Literal, Union

import actfast
import numpy as np
import polars as pl

from wristpy.core import models


def read_watch_data(file_name: Union[pathlib.Path, str]) -> models.WatchData:
    """Read watch data from a file.

    Currently supported watch types are Actigraph .gt3x and GeneActiv .bin.
    Assigns the idle_sleep_mode_flag to false unless the watchtype is .gt3x and
    sleep_mode is enabled (based on watch metadata).

    Args:
        file_name: The filename to read the watch data from.

    Returns:
        WatchData class

    Raises:
        ValueError if the file extension is not supported.
        IOError if the file cannot be read using actfast.
    """
    file_type = pathlib.Path(file_name).suffix
    if file_type not in (".gt3x", ".bin"):
        raise ValueError(f"File type {file_type} is not supported.")
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
    if file_type == ".gt3x":
        idle_sleep_mode_flag = (
            data["metadata"]["device_feature_enabled"]["sleep_mode"].lower() == "true"
        )
        time_zone = data["metadata"]["info"]["TimeZone"]
    elif file_type == ".bin":
        time_zone = data["metadata"]["Configuration Info"]["Time Zone"][4:]

    dynamic_range = _extract_dynamic_range(
        metadata=data["metadata"],
        file_type=file_type,  # type: ignore[arg-type]
    )

    return models.WatchData(
        acceleration=measurements["acceleration"],
        lux=measurements.get("light"),
        battery=measurements.get("battery_voltage"),
        capsense=measurements.get("capsense"),
        temperature=measurements.get("temperature"),
        idle_sleep_mode_flag=idle_sleep_mode_flag,
        dynamic_range=dynamic_range,
        time_zone=str(time_zone),
    )


def _extract_dynamic_range(
    metadata: dict, file_type: Literal[".gt3x", ".bin"]
) -> tuple[float, float]:
    """Extract the dynamic range from metadata.

    Args:
        metadata: Metadata subdictionary where accelerometer range values can be found.
        file_type: Accelerometer data file type. Supports .gt3x and .bin.

    Returns:
        A tuple containing the accelerometer range.

    Raises:
        ValueError: If file type is not supported.
    """
    if file_type == ".gt3x":
        return (
            float(metadata.get("info", {}).get("Acceleration Min")),
            float(metadata.get("info", {}).get("Acceleration Max")),
        )
    elif file_type == ".bin":
        range_str = (
            metadata.get("Device Capabilities", {})
            .get("Accelerometer Range")
            .strip()
            .split(" to ")
        )
        return (float(range_str[0]), float(range_str[1]))

    raise ValueError(f"Unsupported file type given: {file_type}")


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
