[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13883191.svg)](https://doi.org/10.5281/zenodo.13883191)

# `wristpy` <img src="https://media.githubusercontent.com/media/childmindresearch/wristpy/refs/heads/feat/issue-56/update-package-metadata/docs/wristpy_logo.png" align="right" width="25%"/>

 A Python package for wrist-worn accelerometer data processing.



[![Build](https://github.com/childmindresearch/wristpy/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/wristpy/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/childmindresearch/wristpy/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/wristpy)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)
[![LGPL--2.1 License](https://img.shields.io/badge/license-LGPL--2.1-blue.svg)](https://github.com/childmindresearch/wristpy/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/wristpy)

Welcome to wristpy, a Python library designed for processing and analyzing wrist-worn accelerometer data. This library provides a set of tools for loading sensor information, calibrating raw accelerometer data, calculating physical activity metrics (ENMO derived) and sleep metrics (angle-Z derived), finding non-wear periods, and detecing sleep periods (onset and wakeup times). Additionally, we provide access to other sensor data that may be recorded by the watch, including; temperature, luminosity, capacitive sensing, battery voltage, and all metadata.

## Supported formats & devices

The package currently supports the following formats:

| Format | Manufacturer | Device | Implementation status |
| --- | --- | --- | --- |
| GT3X | Actigraph | wGT3X-BT | ✅ |
| BIN | GENEActiv | GENEActiv | ✅ |

**Special Note**   
    The `idle_sleep_mode` for Actigraph watches will lead to uneven sampling rates during periods of no motion (read about this [here](https://actigraphcorp.my.site.com/support/s/article/Idle-Sleep-Mode-Explained)). Consequently, this causes issues when implementing wristpy's non-wear and sleep detection. As of this moment, the authors of this pacakge do not take any steps to impute data during these time gaps and would caution to not use data collected with this mode enabled. Of course, users can make use of the readers within wristpy for their own analysis with this type of data.

## Processing pipeline implementation

The main processing pipeline of the wristpy module can be described as follows:

- Data loading: sensor data is loaded using [`actfast`](https://github.com/childmindresearch/actfast), and a `WatchData` object is created to store all sensor data
- Data calibration: A post-manufacturer calibration step can be applied, to ensure that the acceleration sensor is measuring 1*g* force during periods of no motion. There are three possible options: `None`, `gradient`, `ggir`.
- Metrics Calculation: Calculates various metrics on the calibrated data, namely ENMO (euclidean norm , minus one) and angle-Z (angle of acceleration relative to the *x-y* axis).
- Non-wear detection: We find periods of non-wear based on the acceleration data. Specifically, the standard deviation of the acceleration values in a given time window, along each axis, is used as a threshold to decide `wear` or `not wear`.
- Sleep Detection: Using the HDCZ<sup>1</sup> and HSPT<sup>2</sup> algorithms to analyze changes in arm angle we are able to find periods of sleep. We find the sleep onset-wakeup times for all sleep windows detected.
- Physical activity levels: Using the enmo data (aggreagated into epoch 1 time bins, 5 second default) we compute activity levels into the following categories: inactivity, light activity, moderate activity, vigorous activity. The default threshold values have been chosen based on the values presented in the Hildenbrand 2014 study<sup>3</sup>.


## Installation

Install this package from PyPI via :

```sh
pip install wristpy
```

## Quick start

### Using Wristpy through the command-line:
```sh
wristpy /input/file/path.gt3x -o /save/path/file_name.csv -c gradient
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
2. van Hees, V. T., et al. A Novel, Open Access Method to Assess Sleep
            Duration Using a Wrist-Worn Accelerometer. PLoS One 10, e0142533 (2015).
            https://doi.org/10.1371/journal.pone.0142533
3. Hildebrand, M., et al. Age group comparability of raw accelerometer output
            from wrist- and hip-worn monitors. Medicine and Science in
            Sports and Exercise, 46(9), 1816-1824 (2014).
            https://doi.org/10.1249/mss.0000000000000289

