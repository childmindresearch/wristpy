wristpy Documentation
=====================

.. image:: _static/images/wristpy_logo.png
   :alt: wristpy logo
   :align: center
   :width: 200px

|DOI| |Build| |codecov| |Ruff| |stability| |License| |docs|

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.13883190.svg
   :target: https://doi.org/10.5281/zenodo.13883190

.. |Build| image:: https://github.com/childmindresearch/wristpy/actions/workflows/test.yaml/badge.svg?branch=main
   :target: https://github.com/childmindresearch/wristpy/actions/workflows/test.yaml?query=branch%3Amain

.. |codecov| image:: https://codecov.io/gh/childmindresearch/wristpy/branch/main/graph/badge.svg?token=22HWWFWPW5
   :target: https://codecov.io/gh/childmindresearch/wristpy

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff

.. |stability| image:: https://img.shields.io/badge/stability-experimental-orange.svg

.. |License| image:: https://img.shields.io/badge/license-LGPL--2.1-blue.svg
   :target: https://github.com/childmindresearch/wristpy/blob/main/LICENSE

.. |docs| image:: https://img.shields.io/badge/api-docs-blue
   :target: https://childmindresearch.github.io/wristpy

Welcome to wristpy, a Python library designed for processing and analyzing wrist-worn accelerometer data.

This library provides a set of tools for loading sensor information, calibrating raw accelerometer data, calculating various physical activity metrics, finding non-wear periods, and detecting sleep periods (onset and wakeup times). Additionally, we provide access to other sensor data that may be recorded by the watch, including; temperature, luminosity, capacitive sensing, battery voltage, and all metadata.

Quick Start
-----------

**Installation**

.. note::
   **macOS users**: Install ``libomp`` first via ``brew install libomp``

Install wristpy from PyPI::

   pip install wristpy

**Basic Usage**

Single file processing::

   from wristpy.core import orchestrator
   
   results = orchestrator.run(
       input='/path/to/your/file.gt3x',
       output='/path/to/save/file_name.csv',
       calibrator='gradient'
   )

For detailed examples and usage patterns, see the :doc:`tutorial`.

Supported Devices & Formats
---------------------------

wristpy currently supports:

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Format
     - Manufacturer  
     - Device
     - Status
   * - GT3X
     - Actigraph
     - wGT3X-BT
     - ✅ Supported
   * - BIN
     - GENEActiv
     - GENEActiv
     - ✅ Supported

Processing Pipeline
-------------------

The wristpy processing pipeline includes:

- **Data Loading** - Sensor data loaded via `actfast <https://github.com/childmindresearch/actfast>`_
- **Data Calibration** - Post-manufacturer calibration (None, gradient, ggir options)
- **Metrics Calculation** - ENMO, MAD, Actigraph counts, MIMS, angle-Z
- **Non-wear Detection** - Multiple algorithms including CTA and DETACH
- **Sleep Detection** - HDCZ and HSPT algorithms for sleep onset/wakeup detection
- **Physical Activity Levels** - Classification into inactive, light, moderate, vigorous categories

For complete details, see the :doc:`tutorial`.

Getting Help
------------

- **GitHub Repository**: `childmindresearch/wristpy <https://github.com/childmindresearch/wristpy>`_
- **Issues & Bug Reports**: `GitHub Issues <https://github.com/childmindresearch/wristpy/issues>`_
- **Contributing**: See our :doc:`development` guide

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorial

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/wristpy.core
   api/wristpy.io
   api/wristpy.processing



.. toctree::
   :maxdepth: 1
   :caption: Development

   development

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`