"""Comparing GGIR outputs to wristpy outputs for epoch 1."""

import numpy as np
import polars as pl
import pathlib
from wristpy.common.data_model import OutputData


def load_ggir_output(filepath:pathlib.Path)-> pl.Dataframe:
    """
    """

def compare(ggir_dataframe: pl.DataFrame, wristpy_dataframe: OutputData)-> pl.DataFrame:
    """Compares a wristpy and ggir dataframe.

    Args:
        Two dataframes, one from each package

    Returns:
        A error/difference graph
    """


    error = np.abs(wristpy_dataframe- ggir_dataframe)

    return error
