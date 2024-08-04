
import pytest
from datetime import datetime, timedelta
from typing import List
import numpy as np
from wristpy.processing.metrics import (
    SleepWindow,
    time_to_seconds,
    interdaily_stability_metric,
    sleep_regularity_index,
    social_jet_lag_metric,
    std_dev_metric,
)
import pytest
from datetime import datetime, timedelta
from wristpy.processing.metrics import SleepWindow

@pytest.fixture(autouse=True)
def sleep_window_test_cases():
    base_date = datetime(2023, 5, 1)
    
    return {
        "single_day": {
            "no_sleep": [],
            "whole_day": [
                SleepWindow(onset=base_date, wakeup=base_date + timedelta(days=1))
            ],
            "normal_sleep": [
                SleepWindow(onset=base_date + timedelta(hours=22), wakeup=base_date + timedelta(days=1, hours=6))
            ],
            "two_periods": [
                SleepWindow(onset=base_date + timedelta(hours=22), wakeup=base_date + timedelta(days=1, hours=2)),
                SleepWindow(onset=base_date + timedelta(days=1, hours=3, minutes=15), wakeup=base_date + timedelta(days=1, hours=7))
            ],
            "three_periods": [
                SleepWindow(onset=base_date + timedelta(hours=22), wakeup=base_date + timedelta(days=1, hours=1)),
                SleepWindow(onset=base_date + timedelta(days=1, hours=2, minutes=15), wakeup=base_date + timedelta(days=1, hours=5)),
                SleepWindow(onset=base_date + timedelta(days=1, hours=6, minutes=15), wakeup=base_date + timedelta(days=1, hours=8))
            ]
        },
        "two_days": {
            "non_overlapping": [
                SleepWindow(onset=base_date + timedelta(hours=22), wakeup=base_date + timedelta(days=1, hours=6)),
                SleepWindow(onset=base_date + timedelta(days=1, hours=23), wakeup=base_date + timedelta(days=2, hours=7))
            ],
            "completely_overlapping": [
                SleepWindow(onset=base_date + timedelta(hours=22), wakeup=base_date + timedelta(days=1, hours=6)),
                SleepWindow(onset=base_date + timedelta(days=1, hours=22), wakeup=base_date + timedelta(days=2, hours=6))
            ],
            "partially_overlapping": [
                SleepWindow(onset=base_date + timedelta(hours=22), wakeup=base_date + timedelta(days=1, hours=6)),
                SleepWindow(onset=base_date + timedelta(days=1, hours=23), wakeup=base_date + timedelta(days=2, hours=7))
            ],
            "non_consecutive": [
                SleepWindow(onset=base_date + timedelta(hours=22), wakeup=base_date + timedelta(days=1, hours=6)),
                SleepWindow(onset=base_date + timedelta(days=3, hours=23), wakeup=base_date + timedelta(days=4, hours=7))
            ],
            "weekday_weekend": [
                SleepWindow(onset=base_date + timedelta(hours=22), wakeup=base_date + timedelta(days=1, hours=6)),
                SleepWindow(onset=base_date + timedelta(days=5, hours=23), wakeup=base_date + timedelta(days=6, hours=8))
            ]
        },
        "five_days": {
            "mixed_consecutive": [
                SleepWindow(onset=base_date + timedelta(hours=22), wakeup=base_date + timedelta(days=1, hours=6)),
                SleepWindow(onset=base_date + timedelta(days=1, hours=23), wakeup=base_date + timedelta(days=2, hours=7)),
                SleepWindow(onset=base_date + timedelta(days=4, hours=21), wakeup=base_date + timedelta(days=5, hours=5)),
                SleepWindow(onset=base_date + timedelta(days=5, hours=22), wakeup=base_date + timedelta(days=6, hours=6)),
                SleepWindow(onset=base_date + timedelta(days=7, hours=23), wakeup=base_date + timedelta(days=8, hours=7))
            ]
        }
    }
