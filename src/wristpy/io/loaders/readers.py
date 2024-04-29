"""Function to read accelermoeter data from a file."""

import pathlib

import actfast
import polars as pl

from wristpy.common.data_model import InputData, WatchData


def read_watch_data(filename: pathlib.Path | str) -> WatchData:
    """Read watch data from a file.

    This function selects the correct loader based on the file extension.
    Returns error if none of the above.

    Args:
        filename: The filename to read the watch data from.

    Returns:
        WatchData: The watch data.
    """
    filename = pathlib.Path(filename)
    if filename.suffix == ".gt3x":
        input_data = gt3x_loader(filename)
    elif filename.suffix == ".bin":
        input_data = geneActiv_loader(filename)
    else:
        raise ValueError(f"Unsupported file extension: {filename.suffix}")

    return input_data


def gt3x_loader(
    path: pathlib.Path,
) -> InputData:
    """Load input data from .gt3x file using actfast.

        This loads the acceleration, lux, battery voltage, and capsense data.

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

    # light dataframe, load light data +light time
    lux_values = subject1["timeseries"]["lux"]["lux"]
    lux_time = subject1["timeseries"]["lux"]["datetime"]
    lux_datetime = pl.from_epoch(lux_time, time_unit="ns").alias("time")
    lux_df = pl.DataFrame({"lux": lux_values, "time": lux_datetime})

    # battery voltage dataframe, load battery data + battery time
    battery_data = subject1["timeseries"]["battery_voltage"]["battery_voltage"]
    battery_time = subject1["timeseries"]["battery_voltage"]["datetime"]
    battery_datetime = pl.from_epoch(battery_time, time_unit="ns").alias("time")
    battery_df = pl.DataFrame(
        {"battery_voltage": battery_data, "time": battery_datetime}
    )

    # capsense dataframe, load capsense data + capsense time
    capsense_data = subject1["timeseries"]["capsense"]["capsense"]
    capsense_time = subject1["timeseries"]["capsense"]["datetime"]
    capsense_datetime = pl.from_epoch(capsense_time, time_unit="ns").alias("time")
    capsense_df = pl.DataFrame({"cap_sense": capsense_data, "time": capsense_datetime})

    return InputData(
        acceleration=acceleration,
        sampling_rate=sampling_rate,
        time=time_actfast,
        lux_df=lux_df,
        battery_df=battery_df,
        capsense_df=capsense_df,
    )


def geneActiv_loader(
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
