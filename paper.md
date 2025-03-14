---
title: 'wristpy: Fast, User-Friendly Python Processing of Wrist-worn Accelerometer Data'
tags:
  - Python
  - accelerometer
  - physical activity
  - sleep detection
  - wearable sensors
authors:
  - name: Adam Santorelli
    affiliation: 1
  - name: Freymon Perez
    affiliation: 1
  - name: Reinder Vos de Wael
    affiliation: 1
  - name: Florian Rupprecht
    affiliation: 1
  - name: John Vito d'Antonio-Bertagnolli
    affiliation: 1
  - name: Stan Colcombe
    affiliation: "1, 2"
  - name: Alexandre Franco
    affiliation: "1, 3"
  - name: Gregory Kiar
    affiliation: 1

affiliations:
  - name: Child Mind Institute, New York, USA
    index: 1
  - name: NKI or something
    index: 2
  - name: Third option
    index: 3
date: 14 March 2025
bibliography: paper.bib
---

# Summary

`wristpy` is an open-source Python package designed to streamline the processing and analysis of wrist-worn accelerometer data. The package has been developed in a modular framework to process actigraphy data from both GENEActiv and Actigraph watches, while supporting the extension for other wearable standards in the future, and is supported across multiple platforms (Windows, MacOS, and Ubuntu). This software can be accessed and used through either a command-line interface (CLI) or as an importable Python module. The main processing pipeline will generate outputs in either a `.csv` or `.parquet` format. The output contains timeseries data at a user-chosen temporal resolution, including: physical activity metrics, angle-z, ENMO, sleep onset and wake up times, non-wear period identification, and physical activity classification.


`wristpy` has been designed modularly, which allows for incremental improvements of bottlenecks in the processing pipeline —  such as the independent creation and adoption of a Rust-based reader for actigraphy data that dramatically improves scalability over the vendor-provided reader. With `wristpy`, researchers can directly access a variety of functions that allow them to read specific sensor data stored on the watch, calculate a variety of physical activity metrics, or compare various non-wear or sleep detection algorithms, or take advantage of them together in a 1-click end-to-end preprocessing pipeline.


# Statement of Need

Wearable accelerometers are increasingly used to measure both physical activity and sleep patterns as they provide a simple, effective, non-invasive, and low-cost alternative to clinical observation, allowing for large scale data to be acquired in realistic scenarios. While several commercial solutions exist, such as consumer-grade products providing aggregate data (e.g., FitBit) or research-focused proprietary software (e.g., such as those from Actigraph and ActivInsights), a need exists for device-agnostic open-source tools that can process raw accelerometer data with transparency, reproducibility, and scalability, both as a base for evaluation and methodological advancement..

Several open-source packages have been developed, namely GGIR [@van_hees_ggir_2025], Scikit Digital Health (`skdh`) [@adamowicz_scikit_2022], and pyActigraphy [@hammad_pyactigraphy_2021]. GGIR is one of the most commonly used software packages to process and analyze actigraphy data from various watches and has been cited in hundreds of publications over the past decade. `skdh` is a Python package that can process data from GENEActiv watches to compute physical activity metrics, sleep detection based on [@christakis_sleeppy_2019], novel wear detection [@vert_detecting_2022], gait detection, and numerous other functions for both processing and analyzing accelerometer data. 

While tools exist that can read Actigraph data (GGIR), or detect multiple sleep windows per night (`skdh`), our requirements to flexibly combine data from various watches with various metrics precluded us from using these. An additional limiting factor of each of these libraries is their inability to extensibly support data from other sensors, including those on the devices they currently support (such as skin conductance and light). `wristpy` addresses needs by providing a development environment and utilities that allow it to:
- Process raw accelerometer data from various watch manufacturers, including both GENEActiv and Actigraph watches (with plans to support more research grade devices in future development).
- Access all sensors and metadata from these watches.
- Support modular processing algorithms that can be easily extended while enforcing strict code quality guidelines, exhaustive test coverage, and documentation.
- Be run in a 1-click fashion with an easy to use interface that requires no domain specific knowledge. 
- Allows batch processing for entire directories of data (e.g. a directory containing all data from a specific protocol can be processed in a 1-click fashion)
- Provide a suite of functionality for more advanced users to request specific outputs (choice of physical activity metrics, user-defined temporal resolution, activity thresholds, different non-wear algorithms).

Critically, `wristpy` leverages the history and experience of each of these tools, and supports the algorithms they have pioneered. While the need for `wristpy` was apparent due to the noted limitations, it cannot be overstated how influential these prior toolboxes have been in its construction.


# Processing Pipeline

`wristpy` provides the following key functionalities within the main processing pipeline:

- Data loading using `actfast` [@florian_rupprecht_childmindresearchactfast_2025] a Rust-based reader that provides up to nanosecond temporal resolution with direct access to all sensor information on the watch and key metadata information.
- Post-manufacturer calibration to remove any bias in the device. The two primary methods are a direct minimization method and the default method in GGIR [@van_hees_autocalibration_2014].
- Calculation of essential physical activity metrics such as ENMO (Euclidean Norm Minus One), angle-Z (orientation of the watch relative to the x-y axis), MAD (mean amplitude deviation) [@aittasalo_mean_2015], and Actigraph activity counts [@neishabouri_quantification_2022]. With current development to include physical activity metrics such as MIMS (monitor independent motor summary unit) [@john_open-source_2019].
- Implementation of validated algorithms for on-body wear detection namely from GGIR [@van_hees_ggir_2025], the combined temperature and acceleration algorithm from [@Zhou2014], the `skdh` DETACH algorithm, and ensemble nonwear classification (PR currently under review).
- Sleep period detection: following the default parameters from GGIR, we use the HDCZ [@van_hees_estimating_2018] and HSPT algorithms [@van_hees_novel_2015] to find sleep period candidates. Unlike the implementation within GGIR, we output all viable sleep window candidates across the entire available data. Furthermore, functionality is currently being implemented to remove overlap between sleep periods and non-wear times, eliminating the need for the researcher to verify or remove those overlaps manually.
- Physical activity level classification: Categorize periods of physical activity as `inactive`, `light`, `moderate`, or `vigorous` activity, based on established thresholds for ENMO [@hildebrand_age_2014] or MAD, or custom user-defined thresholds.

# Acknowledgements 
Financial and scientific support has been provided by Dr Michael P. Milham. Financial support has been provided by the California Department of Health Care Services (DHCS) as part of the Children and Youth Behavioral Health Initiative (CYBHI). We would also like to thank Dr. Michelle Freund, Dr. Vadim Zipunnikov, Dr. Andrew Leroux, and Dr. Kathleen R. Merikangas and her team at the NIMH, for their technical and administrative support.


# References
