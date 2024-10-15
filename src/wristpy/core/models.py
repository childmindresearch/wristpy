"""Internal data model."""

from typing import Optional

import numpy as np
import polars as pl
from pydantic import BaseModel, field_validator


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
