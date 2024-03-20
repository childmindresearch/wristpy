"""Internal data model."""

import pathlib
from dataclasses import dataclass, field

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
    sampling_rate: int
    time: pl.DataFrame


@dataclass
class OutputData:
    """Processed actigraphy data and metadata.

    This class should provide access to all processed data.
    It is mutated during processing.
    """

    cal_acceleration: pl.DataFrame = field(default_factory=pl.DataFrame)
    anglez: pl.DataFrame = field(default_factory=pl.DataFrame)
    anglez_epoch1: pl.DataFrame = field(default_factory=pl.DataFrame)
    cal_error_end: float = 0.0
    cal_error_start: float = 0.0
    enmo: pl.DataFrame = field(default_factory=pl.DataFrame)
    enmo_epoch1: pl.DataFrame = field(default_factory=pl.DataFrame)
    offset: float = 0.0
    sampling_rate: float = 0.0
    scale: float = 0.0
    time: pl.DataFrame = field(default_factory=pl.DataFrame)
    time_epoch1: pl.DataFrame = field(default_factory=pl.DataFrame)
