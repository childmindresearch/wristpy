# MIMS Example Data Documentation

This document provides details on how the test data for the MIMS (Monitor Independent Movement Summary) metric and its subfunctions were generated.


## Data Generation Process

The MIMS algorithm consists of 5 main steps:
* Interpolation
* Extrapolation
* Bandpass Filter
* Aggregation
* Truncation

Each major step has associated test data that was used to validate the wristpy implementation. Test data was gathered using an actigraph device model: wGT3XBT which has a dynamic range of
±8

### 1. Interpolation Test data

The interpolation test data was generated using the following R script and saved as 'actigraph_interpolation_r_version.csv' :

```r
library(MIMSunit)
raw_data <- read.gt3x("~/Github/wristpy/tests/sample_data/example_actigraph.gt3x")
accel_df <- as.data.frame(raw_data)
interpolate_df <- MIMSunit::interpolate_signal(df = accel_df, method = "spline_natural", sr = 100)
write.csv(interpolate_df, "actigraph_interpolation_r_version.csv", row.names = FALSE)
```

### 1. Interpolation Test data

The interpolation test data was generated using the following R script and saved as 'actigraph_interpolation_r_version.csv' :

```r
library(MIMSunit)
raw_data <- read.gt3x("~/Github/wristpy/tests/sample_data/example_actigraph.gt3x")
accel_df <- as.data.frame(raw_data)
interpolate_df <- MIMSunit::interpolate_signal(df = accel_df, method = "spline_natural", sr = 100)
write.csv(interpolate_df, "actigraph_interpolation_r_version.csv", row.names = FALSE)
```

### 2. Bandpass Filter Test data

The Bandpass Filter test data was generated using the following R script and saved as 'mims_butterworth_filtered_data.csv' :

```r
library(MIMSunit)
raw_data <- read.gt3x("~/Github/wristpy/tests/sample_data/example_actigraph.gt3x")
accel_df <- as.data.frame(raw_data)
interpolate_df <- MIMSunit::interpolate_signal(df = accel_df, method = "spline_natural", sr = 100)
filtered_df <- MIMSunit::iir(interpolate_df, sr = 100, cutoff_freq = c(0.2, 5), type = 'pass', filter_type = 'butter')
write.csv(filtered_df, "mims_butterworth_filtered_data.csv", row.names = FALSE)
```

### 3. Aggregation Test data

The Aggregation test data was generated using the following R script and saved as 'aggregation_r_version.csv' :

```r
library(MIMSunit)
raw_data <- read.gt3x("~/Github/wristpy/tests/sample_data/example_actigraph.gt3x")
accel_df <- as.data.frame(raw_data)
interpolate_df <- MIMSunit::interpolate_signal(df = accel_df, method = "spline_natural", sr = 100)
aggregated_df <- <- MIMSunit::aggregate_for_mims( interpolate_df,  epoch = "1 min",  method = "trapz", rectify = TRUE)
write.csv(aggredgated_df, "aggregation_r_version.csv", row.names = FALSE)
```

## References
[1] John D, Tang Q, Albinali F, Intille S. An Open-Source Monitor-Independent Movement Summary for Accelerometer Data Processing. J Meas Phys Behav. 2019 Dec;2(4):268-281. doi: 10.1123/jmpb.2018-0068. PMID: 34308270; PMCID: PMC8301210.
