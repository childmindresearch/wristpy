[![DOI](https://zenodo.org/badge/657341621.svg)](https://zenodo.org/doi/10.5281/zenodo.10383685)

# Wristpy: Wrist-Worn Accelerometer Data Processing <img src="logo.png" align="right" width="25%"/>




[![Build](https://github.com/childmindresearch/wristpy/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/wristpy/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/childmindresearch/wristpy/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/wristpy)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)
[![LGPL--2.1 License](https://img.shields.io/badge/license-LGPL--2.1-blue.svg)](https://github.com/childmindresearch/wristpy/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/wristpy)

Welcome to wristpy, a Python library designed for processing and analyzing wrist-worn accelerometer data. This library provides a set of tools for calibrating raw accelerometer data, calculating physical activity metrics (ENMO derived) and sleep metrics (angle-Z derived), finding non-wear periods, and detecing sleep periods (onset and wakeup times). Additionally, we provide access to other sensor dat that may be recorded by the watch, including; temperature, luminosity, capacitive sensing, battery voltage, and all metadata.

## Supported formats & devices

The package currently supports the following formats:

| Format | Manufacturer | Device | Implementation status |
| --- | --- | --- | --- |
| GT3X | Actigraph | wGT3X-BT | ✅ |
| BIN | GENEActiv | GENEActiv | ✅ |


## Features

- GGIR Calibration: Applies the GGIR calibration procedure to raw accelerometer data.
- Metrics Calculation: Calculates various metrics on the calibrated data, namely ENMO (euclidean norm , minus one) and angle-Z (angle of acceleration relative to the *x-y* axis).
- Physical activity levels: Using the enmo data (aggreagated into epoch 1 time bins, 5 second default) we compute activity levels into the following categories: inactivity, light activity, moderate activity, vigorous activity. 
- Non-wear detection: We find periods of non-wear based on the acceleration data. 
- Sleep Detection: Using the HDCZ<sup>1</sup> and HSPT<sup>2</sup> algorithms to analyze changes in arm angle to find periods of sleep. We find the sleep onset-wakeup times for all sleep windows detected.


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

### Using Wristpy through the command-line:
```sh
python wristpy /input/file/path.gt3x -o /save/path/file_name.csv -c gradient
```

### Using Wristpy through a python script or notebook:

```Python

from wristpy.core import orchestrator

# Define input file path and output location
# Support for saving as .csv and .parquet
input_path = '/path/to/your/file.gt3x'
output_path = '/path/to/save/file_name.csv'

# Run the orchestrator
results = orchestrator.run(
    input=input_path,
    output=output_path,
    calibrator='gradient',  # Choose between 'ggir', 'gradient', or 'none'
)

#Data availble in results object
enmo = results.enmo
anglez = results.anglez
physical_activity_levels = results.physical_activity_levels
nonwear_array = results.nonwear_epoch
sleep_windows = results.sleep_windows_epoch
```

## References
1. van Hees, V.T., Sabia, S., Jones, S.E. et al. Estimating sleep parameters
              using an accelerometer without sleep diary. Sci Rep 8, 12975 (2018).
              https://doi.org/10.1038/s41598-018-31266-z
2. van Hees, V. T. et al. A Novel, Open Access Method to Assess Sleep
            Duration Using a Wrist-Worn Accelerometer. PLoS One 10, e0142533 (2015).
            https://doi.org/10.1371/journal.pone.0142533

