"""Internal data model."""

import pathlib
from typing import Optional

import numpy as np
import polars as pl
import pydantic
from pydantic import BaseModel, field_validator

from wristpy.core import config, exceptions

VALID_FILE_TYPES = (".csv", ".parquet")

logger = config.get_logger()


class Measurement(BaseModel):
    """A single measurement of a sensor and its corresponding time."""

    measurements: np.ndarray
    time: pl.Series

    @classmethod
    def from_data_frame(cls, data_frame: pl.DataFrame) -> "Measurement":
        """Creates a measurement from a Polars DataFrame.

        Args:
            data_frame: The Polars DataFrame, must have a time column. All
                non-time columns will be used as the 'measurements' input.
        """
        return Measurement(
            measurements=data_frame.drop("time").to_numpy().squeeze(),
            time=data_frame["time"],
        )

    def lazy_frame(self) -> pl.LazyFrame:
        """Converts the measurement to a LazyFrame.

        Returns:
            The Measurement as a LazyFrame. The time property will have column name
                'time'. Other column names should not be relied upon.
        """
        return pl.concat(
            [
                pl.LazyFrame(self.measurements),
                pl.LazyFrame({"time": self.time}).set_sorted("time"),
            ],
            how="horizontal",
        )

    class Config:
        """Config to allow for ndarray as input."""

        arbitrary_types_allowed = True

    @field_validator("measurements")
    def validate_measurements_not_empty(cls, v: np.ndarray) -> np.ndarray:
        """Validate that the measurements array is not empty.

        Args:
            cls: The class.
            v: The measurements array to validate.

        Returns:
            v: The measurements array if it is not empty.

        Raises:
            ValueError: If the measurements array is empty.
        """
        if v.size == 0:
            raise ValueError("measurements array must not be empty")
        return v

    @field_validator("time")
    def validate_time(cls, v: pl.Series) -> pl.Series:
        """Validate the time series.

        Check that the time series is a datetime series, contains only unque entries,
        and is sorted.

        Args:
            cls: The class.
            v: The time series to validate.

        Returns:
            v: The time series if it is valid.

        Raises:
            ValueError: If the time series is not a datetime series or is not sorted,
            or is empty.
        """
        if not isinstance(v.dtype, pl.datatypes.Datetime):
            raise ValueError("Time must be a datetime series")
        if not v.is_unique().all():
            raise ValueError("Time series must contain unique entries")
        if not v.is_sorted():
            raise ValueError("Time series must be sorted")
        if v.is_empty():
            raise ValueError("Time series cannot be empty")
        return v


class WatchData(BaseModel):
    """Watch data that is read off the device.

    This class should provide access to all raw input data.
    It must not be mutated during processing.
    """

    acceleration: Measurement
    lux: Optional[Measurement] = None
    battery: Optional[Measurement] = None
    capsense: Optional[Measurement] = None
    temperature: Optional[Measurement] = None

    @field_validator("acceleration")
    def validate_acceleration(cls, v: Measurement) -> Measurement:
        """Validate the acceleration data.

        Ensure that the acceleration data is a 2D array with 3 columns.

        Args:
            cls: The class.
            v: The acceleration data to validate.

        Returns:
            v: The acceleration data if it is valid.

        Raises:
            ValueError: If the acceleration data is not a 2D array with 3 columns.
        """
        if v.measurements.ndim != 2 or v.measurements.shape[1] != 3:
            raise ValueError("acceleration must be a 2D array with 3 columns")
        return v


class Results(pydantic.BaseModel):
    """dataclass containing results of orchestrator.run()."""

    enmo: Measurement
    anglez: Measurement
    physical_activity_levels: Measurement
    nonwear_epoch: Measurement
    sleep_windows_epoch: Measurement

    def save_results(self, output: pathlib.Path) -> None:
        """Convert to polars and save the dataframe as a csv or parquet file.

        Args:
            output: The path and file name of the data to be saved. as either a csv or
                parquet files.

        """
        logger.debug("Saving results.")
        self.validate_output(output=output)

        results_dataframe = pl.DataFrame(
            {"time": self.enmo.time}
            | {name: value.measurements for name, value in self}
        )

        if output.suffix == ".csv":
            results_dataframe.write_csv(output, separator=",")
        elif output.suffix == ".parquet":
            results_dataframe.write_parquet(output)
        else:
            raise exceptions.InvalidFileTypeError(
                f"File type must be one of {VALID_FILE_TYPES}"
            )

        logger.debug("results saved in: %s", output)

    @classmethod
    def validate_output(cls, output: pathlib.Path) -> None:
        """Validates that the output path exists and is a valid format.

        Args:
            output: the name of the file to be saved, and the directory it will
                be saved in. Must be a .csv or .parquet file.

        Raises:
            InvalidFileTypeError:If the output file path ends with any extension other
                    than csv or parquet.
        """
        output.parent.mkdir(parents=True, exist_ok=True)

        if output.suffix not in VALID_FILE_TYPES:
            raise exceptions.InvalidFileTypeError(
                f"The extension: {output.suffix} is not supported."
                "Please save the file as .csv or .parquet",
            )
