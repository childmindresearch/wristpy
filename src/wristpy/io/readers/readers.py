"""Function to read accelerometer data from a file."""

import datetime
import pathlib
import re
from typing import Literal, Union

import actfast
import numpy as np
import polars as pl

from wristpy.core import config, models

logger = config.get_logger()


def read_watch_data(
    file_name: Union[pathlib.Path, str], *, allow_duplicates: bool = False
) -> models.WatchData:
    """Read watch data from a file.

    Currently supported watch types are Actigraph .gt3x and GeneActiv .bin.
    Assigns the idle_sleep_mode_flag to false unless the watchtype is .gt3x and
    sleep_mode is enabled (based on watch metadata).

    If requested, removes duplicate timestamps from the data, keeping only unique
    timestamps and their corresponding sensor values.

    There is also support for .csv files that have been processed with ActiGraph
    software and exported into csv format as part of the HBN Actigraphy data release.
    For these files, we assume linearly sampled data based on the provided sampling
    rate in the metadata, the timezone is set to New York,
    and idle_sleep_mode_flag is set to False.

    Args:
        file_name: The filename to read the watch data from.
        allow_duplicates: Whether to allow duplicate timestamps in the data. If
            False, duplicate timestamps will raise a ValueError in the
            Measurement validation phase. If set to True, we will keep only the
            unique timestamps and the associated sensor values. The first occurrence
            of each timestamp is kept.
            Default is False.

    Returns:
        WatchData class

    Raises:
        ValueError if the file extension is not supported.
        IOError if the file cannot be read using actfast.
    """
    file_type = pathlib.Path(file_name).suffix
    if file_type not in (".gt3x", ".bin", ".csv"):
        raise ValueError(f"File type {file_type} is not supported.")
    if file_type == ".csv":
        acceleration_data, metadata = _read_actigraph_csv(pathlib.Path(file_name))

        n_samples = len(acceleration_data)
        timestamps = [
            metadata["start_datetime"]
            + datetime.timedelta(seconds=i / metadata["sampling_rate"])
            for i in range(n_samples)
        ]
        time_series = pl.Series(timestamps).cast(pl.Datetime("ns"))

        acceleration_measurement = models.Measurement(
            measurements=acceleration_data, time=time_series
        )

        return models.WatchData(
            acceleration=acceleration_measurement,
            lux=None,
            battery=None,
            capsense=None,
            temperature=None,
            idle_sleep_mode_flag=False,
            dynamic_range=(-8, 8),
            time_zone="America/New_York",
        )
    try:
        data = actfast.read(file_name, lenient=True)
        warnings = data.get("warnings", [])
        if warnings:
            logger.warning(
                f"Recovered partial data for {file_name} "
                f"with {len(warnings)} warnings."
            )
    except Exception as e:
        raise IOError(f"Error reading file: {e}. File type is unsupported.") from e

    measurements: dict[str, models.Measurement] = {}

    for timeseries in data["timeseries"].values():
        time = unix_epoch_time_to_polars_datetime(timeseries["datetime"])
        if allow_duplicates:
            logger.info(
                "Keeping only unique timestamps as requested. "
                "Please note that there may have been duplicate timestamps, "
                "which is indicative of sensor malfunction."
            )
            unique_time_indices = time.arg_unique()
            time = time.gather(unique_time_indices)

        for sensor_name, sensor_values in timeseries.items():
            if not isinstance(sensor_values, np.ndarray):
                continue
            if allow_duplicates:
                sensor_values = sensor_values[unique_time_indices.to_numpy()]

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


def _read_actigraph_csv(filepath: pathlib.Path) -> tuple[np.ndarray, dict]:
    """Read ActiGraph CSV file with metadata headers.

    This helper function is used to read raw actigraphy data that has been processed
    with ActiGraph software and exported into csv format as part of the
    HBN Actigraphy data release.

    We assume linearly sampled data based on the provided sampling rate in the metadata,
    the timezone is set to New York, and idle_sleep_mode_flag is set to False.
    If no sampling rate is provided, we default to 60 Hz.

    Args:
        filepath: Path to the ActiGraph CSV file

    Returns:
        Tuple of (acceleration_data, metadata).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [f.readline() for _ in range(12)]

    hz_match = re.search(r"at (\d+) Hz", lines[0])
    sampling_rate = int(hz_match.group(1)) if hz_match else int(60)

    start_time = lines[2].strip().split()[-1]
    start_date = lines[3].strip().split()[-1]
    start_datetime = datetime.datetime.strptime(
        f"{start_date} {start_time}", "%d/%m/%Y %H:%M:%S"
    )

    data = pl.read_csv(
        filepath,
        skip_rows=12,
        has_header=False,
        schema_overrides={
            "Accelerometer_X": pl.Float64,
            "Accelerometer_Y": pl.Float64,
            "Accelerometer_Z": pl.Float64,
        },
    ).to_numpy()

    metadata = {"sampling_rate": sampling_rate, "start_datetime": start_datetime}

    return data, metadata
