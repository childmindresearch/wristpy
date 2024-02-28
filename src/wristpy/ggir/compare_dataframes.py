"""Comparing GGIR outputs to wristpy outputs for epoch 1."""

from matplotlib import pyplot as plt 

import pathlib

import polars as pl

from wristpy.common.data_model import OutputData


def load_ggir_output(filepath:pathlib.Path)-> pl.Dataframe:
    """
    """

def compare(ggir_dataframe: pl.DataFrame, wristpy_dataframe: OutputData)-> pl.DataFrame:
    """Compares a wristpy and ggir dataframe.

    Args:
        ggir_dataframe:
            The GGIR derived dataframe to be used to calculate difference between GGIR 
            and wristpy outputs.
        wristpy_dataframe:
            The wristpy OutputData object to be used to calculate difference between 
            GGIR and wristpy outputs.
        

    Returns:
        A error/difference graph calculated by taking the difference between 
        the wristpy OutputData and GGIR dataframe
    """
    enmo_error = wristpy_dataframe.enmo_epoch1 - ggir_dataframe.enmo
    anglez_error = wristpy_dataframe.anglez_epoch1 - ggir_dataframe.angelz

    return enmo_error, anglez_error



