[![DOI](https://zenodo.org/badge/657341621.svg)](https://zenodo.org/doi/10.5281/zenodo.10383685)

# Wristpy: Wrist-Worn Accelerometer Data Processing



[![Build](https://github.com/childmindresearch/wristpy/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/wristpy/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/childmindresearch/wristpy/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/wristpy)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)
[![LGPL--2.1 License](https://img.shields.io/badge/license-LGPL--2.1-blue.svg)](https://github.com/childmindresearch/wristpy/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/wristpy)

Welcome to wristpy, a Python library designed for processing and analyzing wrist-worn accelerometer data. This library provides a set of tools for calibrating raw accelerometer data, calculating physical activity metrics (ENMO derived) and sleep metrics (angle-Z derived), finding non-wear periods, and proividing the additional available metadata (temperature, lux, battery voltage, etc.). 


## Features

- GGIR Calibration: Applies the GGIR calibration procedure to raw accelerometer data.
- Non-Movement Identification: Identifies periods of non-movement based on a rolling standard deviation threshold.
- Metrics Calculation: Calculates various metrics on the calibrated data, namely ENMO (euclidean norm , minus one) and angle-Z (angle of acceleration relative to the *x-y* axis).
- All metrics and raw data are provided in an output class, with the calculated metrics downsampled to a fixed epoch resolution of 5s.


## Installation

Install this package via :

```sh
pip install wristpy
```

Or get the newest development version via:

```sh
pip install git+https://github.com/childmindresearch/wristpy
```

## Quick start

Here is an example on how to use wristpy to process .gt3x files collected from Actigraph and save the resulting output data to a .csv file. A similar process can be used with the .bin files from GENEActiv.

```Python

#loading the prerequisite modules
import wristpy
from wristpy.common.data_model import OutputData
from wristpy.io.loaders import gt3x
from wristpy.ggir import calibration, metrics_calc

#set the paths to the raw data and the desired output path
file_name = '/path/to/your/file.gt3x'
output_path = '/path/to/your/output/file.csv'
test_config = wristpy.common.data_model.Config(file_name, output_path)

#load the acceleration data
test_data = gt3x.load_fast(test_config.path_input)

#calibrate the data
test_output = calibration.start_ggir_calibration(test_data)

#compute some desired metrics
metrics_calc.calc_base_metrics(test_output)
metrics_calc.calc_epoch1_metrics(test_output)
metrics_calc.calc_epoch1_raw(test_output)
metrics_calc.set_nonwear_flag(test_output, 900)
metrics_calc.calc_epoch1_light(test_data, test_output)
metrics_calc.calc_epoch1_battery(test_data, test_output)
output_data_csv = pl.DataFrame(
        {
            "time": test_output.time_epoch1,
            "X": test_output.accel_epoch1["X_mean"],
            "Y": test_output.accel_epoch1["Y_mean"],
            "Z": test_output.accel_epoch1["Z_mean"],
            "enmo": test_output.enmo_epoch1,
            "anglez": test_output.anglez_epoch1,
            "Non-wear Flag": test_output.non_wear_flag_epoch1,
            "light": test_output.lux_epoch1,
            "battery voltage": test_output.battery_upsample_epoch1,
        }
    )

#save the output to .csv
output_data_csv.write_csv(output_file_path)
```

## Links or References


