"""Test the orchestrator.py module."""

import datetime
import pathlib

import numpy as np
import polars as pl
import pytest

from wristpy.core import models, orchestrator
from wristpy.processing import analytics


def test_format_sleep() -> None:
    """Test sleep formatter."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_datetime_list = [dummy_date + datetime.timedelta(seconds=i) for i in range(5)]
    dummy_measure = models.Measurement(
        measurements=np.ones(5), time=pl.Series(dummy_datetime_list)
    )
    sleep_window_1 = analytics.SleepWindow(
        onset=dummy_date, wakeup=dummy_date + datetime.timedelta(seconds=1)
    )
    sleep_window_2 = analytics.SleepWindow(
        onset=dummy_date + datetime.timedelta(seconds=3),
        wakeup=dummy_date + datetime.timedelta(seconds=4),
    )

    sleep_array = orchestrator.format_sleep_data(
        sleep_windows=[sleep_window_1, sleep_window_2], epoch1_measure=dummy_measure
    )

    assert (
        len(sleep_array) == len(dummy_measure.measurements) == len(dummy_measure.time)
    )
    assert np.all(sleep_array == np.array([1, 1, 0, 1, 1]))


def test_format_nonwear() -> None:
    """Test nonwear formatter."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_datetime_list = [
        dummy_date + datetime.timedelta(seconds=i * 5) for i in range(4)
    ]
    dummy_epoch1 = models.Measurement(
        measurements=np.ones(4), time=pl.Series(dummy_datetime_list)
    )
    nonwear_measurement = models.Measurement(
        measurements=np.array([1, 0]),
        time=pl.Series([dummy_date, dummy_date + datetime.timedelta(seconds=10)]),
    )

    non_wear_epoch1 = orchestrator.format_nonwear_data(
        nonwear_data=nonwear_measurement, epoch1_measure=dummy_epoch1
    )

    assert (
        len(non_wear_epoch1) == len(dummy_epoch1.measurements) == len(dummy_epoch1.time)
    )
    assert np.all(non_wear_epoch1 == np.array([1, 1, 0, 0]))


def test_results_to_dataframe_epoch1() -> None:
    """Tests converting results object to polars dataframe."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_measure = models.Measurement(
        measurements=np.random.rand(100),
        time=pl.Series(
            [dummy_date + datetime.timedelta(seconds=i) for i in range(100)]
        ),
    )
    dummy_results = orchestrator.Results(
        enmo=dummy_measure,
        anglez=dummy_measure,
        enmo_epoch1=dummy_measure,
        anglez_epoch1=dummy_measure,
        nonwear_array=dummy_measure,
        sleep_windows=[
            analytics.SleepWindow(
                onset=dummy_date, wakeup=dummy_date + datetime.timedelta(seconds=1)
            )
        ],
        physical_activity_levels=dummy_measure,
        nonwear_epoch1=dummy_measure,
        sleep_windows_epoch1=dummy_measure,
    )

    df_epoch1 = dummy_results._results_to_dataframe()

    assert isinstance(df_epoch1, pl.DataFrame)
    assert "time" in df_epoch1.columns
    assert "enmo_epoch1" in df_epoch1.columns
    assert "anglez_epoch1" in df_epoch1.columns
    assert "nonwear_epoch1" in df_epoch1.columns
    assert "sleep_windows_epoch1" in df_epoch1.columns
    assert "physical_activity_levels" in df_epoch1.columns
    assert "enmo" not in df_epoch1.columns
    assert "anglez" not in df_epoch1.columns


def test_results_to_dataframe_raw() -> None:
    """Tests converting results object to polars dataframe."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_measure = models.Measurement(
        measurements=np.random.rand(100),
        time=pl.Series(
            [dummy_date + datetime.timedelta(seconds=i) for i in range(100)]
        ),
    )
    dummy_results = orchestrator.Results(
        enmo=dummy_measure,
        anglez=dummy_measure,
        enmo_epoch1=dummy_measure,
        anglez_epoch1=dummy_measure,
        nonwear_array=dummy_measure,
        sleep_windows=[
            analytics.SleepWindow(
                onset=dummy_date, wakeup=dummy_date + datetime.timedelta(seconds=1)
            )
        ],
        physical_activity_levels=dummy_measure,
        nonwear_epoch1=dummy_measure,
        sleep_windows_epoch1=dummy_measure,
    )

    df_raw_time = dummy_results._results_to_dataframe(use_epoch1_time=False)

    assert isinstance(df_raw_time, pl.DataFrame)
    assert "time" in df_raw_time.columns
    assert "enmo_epoch1" not in df_raw_time.columns
    assert "anglez_epoch1" not in df_raw_time.columns
    assert "nonwear_epoch1" not in df_raw_time.columns
    assert "sleep_windows_epoch1" not in df_raw_time.columns
    assert "physical_activity_levels" not in df_raw_time.columns
    assert "enmo" in df_raw_time.columns
    assert "anglez" in df_raw_time.columns


@pytest.mark.parametrize(
    "file_name", [pathlib.Path("test_output.csv"), pathlib.Path("test_output.parquet")]
)
def test_save_results(file_name: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Test saving."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_measure = models.Measurement(
        measurements=np.random.rand(100),
        time=pl.Series(
            [dummy_date + datetime.timedelta(seconds=i) for i in range(100)]
        ),
    )
    dummy_results = orchestrator.Results(
        enmo=dummy_measure,
        anglez=dummy_measure,
        enmo_epoch1=dummy_measure,
        anglez_epoch1=dummy_measure,
        nonwear_array=dummy_measure,
        sleep_windows=[
            analytics.SleepWindow(
                onset=dummy_date, wakeup=dummy_date + datetime.timedelta(seconds=1)
            )
        ],
        physical_activity_levels=dummy_measure,
        nonwear_epoch1=dummy_measure,
        sleep_windows_epoch1=dummy_measure,
    )
    output_path = tmp_path / file_name
    epoch1_file = pathlib.Path("test_output_epoch1" + file_name.suffix)
    raw_file = pathlib.Path("test_output_raw_time" + file_name.suffix)

    dummy_results.save_results(output_path)

    assert (tmp_path / epoch1_file).exists()
    assert (tmp_path / raw_file).exists()

    pass


def test_save_invalid_file_type(tmp_path: pathlib.Path):
    """test when a bad extention is given."""
    dummy_date = datetime.datetime(2024, 5, 2)
    dummy_measure = models.Measurement(
        measurements=np.random.rand(100),
        time=pl.Series(
            [dummy_date + datetime.timedelta(seconds=i) for i in range(100)]
        ),
    )
    dummy_results = orchestrator.Results(
        enmo=dummy_measure,
        anglez=dummy_measure,
        enmo_epoch1=dummy_measure,
        anglez_epoch1=dummy_measure,
        nonwear_array=dummy_measure,
        sleep_windows=[
            analytics.SleepWindow(
                onset=dummy_date, wakeup=dummy_date + datetime.timedelta(seconds=1)
            )
        ],
        physical_activity_levels=dummy_measure,
        nonwear_epoch1=dummy_measure,
        sleep_windows_epoch1=dummy_measure,
    )

    with pytest.raises(orchestrator.InvalidFileTypeError):
        dummy_results.save_results(tmp_path / "bad_output.vsc")


def test_run() -> None:
    """Test runner."""
    pass
