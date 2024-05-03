"""Internal data model."""

from typing import Optional

import numpy as np
import polars as pl
from pydantic import BaseModel, field_validator


class Measurement(BaseModel):
    """A single measurement of a sensor and its corresponding time."""

    measurements: np.ndarray
    time: pl.Series

    class Config:
        """Config to allow for ndarray as input."""

        arbitrary_types_allowed = True

    @field_validator("time")
    def validate_time(cls, v: pl.Series) -> pl.Series:
        """Validate the time series.

        Check that the time series is a datetime series and sorted.

        Args:
            cls: The class.
            v: The time series to validate.

        Returns:
            v: The time series if it is valid.

        Raises:
            ValueError: If the time series is not a datetime series or is not sorted.
        """
        if not isinstance(v.dtype, pl.datatypes.Datetime):
            raise ValueError("time must be a datetime series")
        if not v.is_sorted():
            raise ValueError("time series must be sorted")
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
        if len(v.measurements.shape) < 2 or v.measurements.shape[1] != 3:
            raise ValueError("acceleration must be a 2D array with 3 columns")
        return v
