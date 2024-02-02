"""Internal data model."""
import pathlib
from dataclasses import dataclass

import polars as pl


@dataclass
class Config:
    """Configuration for the actigraphy pipeline.

    Contains all user-set parameters.
    """

    path_input: pathlib.Path
    path_output: pathlib.Path


@dataclass
class InputData:
    """Raw actigraphy data and metadata.

    This class should provide access to all raw input data.
    It must not be mutated during processing.
    """

    acceleration: pl.DataFrame
    sampling_rate: float


@dataclass
class OutputData:
    """Processed actigraphy data and metadata.

    This class should provide access to all processed data.
    It is mutated during processing.
    """

    pass
