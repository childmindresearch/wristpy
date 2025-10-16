[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13883190.svg)](https://doi.org/10.5281/zenodo.13883190)

# `wristpy` <img src="https://media.githubusercontent.com/media/childmindresearch/wristpy/main/docs/_static/images/wristpy_logo.png" align="right" width="25%"/>

 A Python package for wrist-worn accelerometer data processing.



[![Build](https://github.com/childmindresearch/wristpy/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/wristpy/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/childmindresearch/wristpy/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/wristpy)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)
[![LGPL--2.1 License](https://img.shields.io/badge/license-LGPL--2.1-blue.svg)](https://github.com/childmindresearch/wristpy/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/wristpy)

Welcome to wristpy, a Python library designed for processing and analyzing wrist-worn accelerometer data. This library provides a set of tools for loading sensor information, calibrating raw accelerometer data, calculating various physical activity metrics, finding non-wear periods, and detecting sleep periods (onset and wakeup times). Additionally, we provide access to other sensor data that may be recorded by the watch, including; temperature, luminosity, capacitive sensing, battery voltage, and all metadata.

## Supported formats & devices

The package currently supports the following formats:

| Format | Manufacturer | Device | Implementation status |
| --- | --- | --- | --- |
| GT3X | Actigraph | wGT3X-BT | ✅ |
| BIN | GENEActiv | GENEActiv | ✅ |

**Special Note**
    The `idle_sleep_mode` for Actigraph watches will lead to uneven sampling rates during periods of no motion (read about this [here](https://actigraphcorp.my.site.com/support/s/article/Idle-Sleep-Mode-Explained)). Consequently, this causes issues when implementing wristpy's non-wear and sleep detection. As of this moment, we fill in the missing acceleration data with the assumption that the watch is perfectly idle in the face-up position (Acceleration vector = [0, 0, -1]). The data is filled in at the same sampling rate as the raw acceleration data. In the special circumstance when acceleration samples are not evenly spaced, the data is resampled to the highest effective sampling rate to ensure linearly sampled data.

## Processing pipeline implementation

The main processing pipeline of the wristpy module can be described as follows:

- **Data loading**: sensor data is loaded using [`actfast`](https://github.com/childmindresearch/actfast), and a `WatchData` object is created to store all sensor data.
- **Data calibration**: A post-manufacturer calibration step can be applied, to ensure that the acceleration sensor is measuring 1*g* force during periods of no motion. There are three possible options: `None`, `gradient`, `ggir`.
- ***Data imputation*** In the special case when dealing with the Actigraph `idle_sleep_mode == enabled`, the gaps in acceleration are filled in after calibration, to avoid biasing the calibration phase.
- **Metrics Calculation**: Calculates various activity metrics on the calibrated data, namely ENMO (Euclidean norm, minus one), MAD (mean amplitude deviation) <sup>1</sup>, Actigraph activity counts<sup>2</sup>, MIMS (monitor-independent movement summary) unit <sup>3</sup>, and angle-Z (angle of acceleration relative to the *x-y* axis).
- **Non-wear detection**: We find periods of non-wear based on the acceleration data. Specifically, the standard deviation of the acceleration values in a given time window, along each axis, is used as a threshold to decide `wear` or `not wear`. Additionally, we can use the temperature sensor, when available, to augment the acceleration data. This is used in the CTA (combined temperature and acceleration) algorithm <sup>4</sup>, and in the DETACH algorithm <sup>5</sup>. Furthermore, ensemble classification of non-wear periods is possible by providing a list (of any length) of non-wear algorithm options.
- **Sleep Detection**: Using the HDCZ<sup>6</sup> and HSPT<sup>7</sup> algorithms to analyze changes in arm angle we are able to find periods of sleep. We find the sleep onset-wakeup times for all sleep windows detected. Any sleep periods that overlap with detected non-wear times are removed, and any remaining sleep periods shorter than 15 minutes (default value) are removed. Additionally, the SIB (sustained inactivity bouts) and the SPT (sleep period time) windows are provided as part of the output to aid in sleep metric post-processing.
- **Physical activity levels**: Using the chosen physical activity metric (aggregated into time bins, 5 second default) we compute activity levels into the following categories: [`inactive`, `light`, `moderate`, `vigorous`]. The threshold values can be defined by the user, while the default values are chosen based on the specific activity metric and the values found in the literature  <sup>8-10</sup>.
- **Data output**: The output results can be saved in `.csv` or `.parquet` data formats, with the run-time configuration parameters saved in a `.json` dictionary.


## Installation

Install the `wristpy` package from PyPI via:

```sh
pip install wristpy
```

## Quick start

`wristpy` provides three flexible interfaces: a command-line tool for direct execution, an importable Python library, and a Docker image for containerized deployment.

### Using Wristpy through the command-line:
#### Run single files:
```sh
wristpy /input/file/path.gt3x -o /save/path/file_name.csv -c gradient
```

#### Run entire directories:
```sh
wristpy /path/to/files/input_dir -o /path/to/files/output_dir -c gradient -O .csv
```

#### For a full list of command line arguments:
```sh
wristpy --help
```


### Using Wristpy through a python script or notebook:

#### Running single files:
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

#Data available in results object
physical_activity_metric = results.physical_activity_metric
anglez = results.anglez
physical_activity_levels = results.physical_activity_levels
nonwear_array = results.nonwear_status
sleep_windows = results.sleep_status
sib_periods = results.sib_periods
spt_periods = results.spt_periods

```
#### Running entire directories:
```Python

from wristpy.core import orchestrator

# Define input file path and output location

input_path = '/path/to/files/input_dir'
output_path = '/path/to/files/output_dir'

# Run the orchestrator
# Specify the output file type, support for saving as .csv and .parquet
results_dict = orchestrator.run(
    input=input_path,
    output=output_path,
    calibrator='gradient',  # Choose between 'ggir', 'gradient', or 'none'
    output_filetype = '.csv'
)


#Data available in dictionary of results.
subject1 = results_dict['subject1']

physical_activity_metric = subject1.physical_activity_metric
anglez = subject1.anglez
physical_activity_levels = subject1.physical_activity_levels
nonwear_array = subject1.nonwear_status
sleep_windows = subject1.sleep_status
sib_periods = subject1.sib_periods
spt_periods = subject1.spt_periods

```

### Using Wristpy Through Docker


1. **Install Docker**: Ensure you have Docker installed on your system. [Get Docker](https://docs.docker.com/get-docker/)

2. **Pull the Docker image**:
   ```bash
   docker pull cmidair/wristpy:main
   ```

3. **Run the Docker image** with your data:
   ```bash
   docker run -it --rm \
     -v "/local/path/to/data:/data" \
     -v "/local/path/to/output:/output" \
     cmidair/wristpy:main
   ```
   Replace `/local/path/to/data` with the path to your input data directory and `/local/path/to/output` with where you want results saved.

   To run a single file, we simply need to modify the mounting structure for the docker call slightly:
    ```bash
    docker run -it --rm \
     -v "/local/path/to/data/file.bin:/data/file.bin" \
     -v "/local/path/to/output:/output" \
     cmidair/wristpy:main
   ```

### Customizing the Pipeline:

The Docker image supports multiple input variables to customize processing. You can set these by simply chaining these inputs as you would for the CLI input:

```bash
docker run -it --rm \
  -v "/local/path/to/data/file.bin:/data/file.bin" \
  -v "/local/path/to/output:/output" \
  cmidair/wristpy:main /data --output /output --epoch-length 5 --nonwear-algorithm ggir --nonwear-algorithm detach --thresholds 0.1 0.2 0.4
```



For more details on available options, see the [orchestrator documentation](https://childmindresearch.github.io/wristpy/wristpy/core/orchestrator.html#run).


## References
1.  Vähä-Ypyä H, Vasankari T, Husu P, Suni J, Sievänen H. A universal, accurate
        intensity-based classification of different physical activities using raw data
        of accelerometer. Clin Physiol Funct Imaging. 2015 Jan;35(1):64-70.
        doi: 10.1111/cpf.12127. Epub 2014 Jan 7. PMID: 24393233.
2.  A. Neishabouri et al., “Quantification of acceleration as activity counts
        in ActiGraph wearable,” Sci Rep, vol. 12, no. 1, Art. no. 1, Jul. 2022,
        doi: 10.1038/s41598-022-16003-x.
3.  John, D., Tang, Q., Albinali, F. and Intille, S., 2019. An Open-Source
        Monitor-Independent Movement Summary for Accelerometer Data Processing. Journal
        for the Measurement of Physical Behaviour, 2(4), pp.268-281.
4.  Zhou S, Hill RA, Morgan K, et al, Classification of accelerometer wear and
        non-wear events in seconds for monitoring free-living physical activityBMJ
        Open 2015; 5:e007447. doi: 10.1136/bmjopen-2014-007447.
5.  A. Vert et al., “Detecting accelerometer non-wear periods using change
        in acceleration combined with rate-of-change in temperature,” BMC Medical
        Research Methodology, vol. 22, no. 1, p. 147, May 2022,
        doi: 10.1186/s12874-022-01633-6.
6. van Hees, V.T., Sabia, S., Jones, S.E. et al. Estimating sleep parameters
              using an accelerometer without sleep diary. Sci Rep 8, 12975 (2018).
              https://doi.org/10.1038/s41598-018-31266-z.
7. van Hees, V. T., et al. A Novel, Open Access Method to Assess Sleep
            Duration Using a Wrist-Worn Accelerometer. PLoS One 10, e0142533 (2015).
            https://doi.org/10.1371/journal.pone.0142533.
8. Hildebrand, M., et al. Age group comparability of raw accelerometer output
            from wrist- and hip-worn monitors. Medicine and Science in
            Sports and Exercise, 46(9), 1816-1824 (2014).
            https://doi.org/10.1249/mss.0000000000000289.
9.  Treuth MS, Schmitz K, Catellier DJ, McMurray RG, Murray DM, Almeida MJ,
        Going S, Norman JE, Pate R. Defining accelerometer thresholds for activity
        intensities in adolescent girls. Med Sci Sports Exerc. 2004 Jul;36(7):1259-66.
        PMID: 15235335; PMCID: PMC2423321.
10. Aittasalo, M., Vähä-Ypyä, H., Vasankari, T. et al. Mean amplitude deviation
        calculated from raw acceleration data: a novel method for classifying the
        intensity of adolescents' physical activity irrespective of accelerometer brand.
        BMC Sports Sci Med Rehabil 7, 18 (2015). https://doi.org/10.1186/s13102-015-0010-0.
