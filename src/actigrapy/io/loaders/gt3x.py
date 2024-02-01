"""Loader for the .gt3x file format."""

import pathlib

import polars as pl
import pygt3x.reader
from wristpy.common.data_model import InputData


def load(
    path: pathlib.Path,
) -> InputData:
    """Load input data from .gt3x file."""
    with pygt3x.reader.FileReader(str(path)) as reader:
        acceleration = pl.from_pandas(reader.to_pandas())
        sampling_rate = reader.info.sample_rate

    return InputData(
        acceleration=acceleration,
        sampling_rate=sampling_rate,
    )
