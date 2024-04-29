"""Internal data model."""

from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import polars as pl
from pydantic import BaseModel


class Measurement(BaseModel):
    """A single measurement of a sensor and its corresponding time."""

    measurements: np.ndarray
    time: pl.Series

    class Config:
        """Config to allow for ndarray as input."""

        arbitrary_types_allowed = True


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
