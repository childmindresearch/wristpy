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
    def validate_time(cls, v):  # noqa: ANN201, D102
        if not isinstance(v.dtype, pl.datatypes.Datetime):
            raise ValueError("time must be a datetime series")
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
