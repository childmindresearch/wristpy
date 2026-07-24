Wristpy Documentation
=====================

.. image:: _static/images/wristpy_logo_light.png
   :alt: wristpy logo light
   :align: center
   :width: 200px
   :class: only-light

.. image:: _static/images/wristpy_logo_dark.png
   :alt: wristpy logo dark
   :align: center
   :width: 200px
   :class: only-dark

|DOI| |Build| |codecov| |Ruff| |stability| |License| |docs|

.. |DOI| image:: https://joss.theoj.org/papers/88f3fdfdf621e967da12a9e8fde4785d/status.svg
   :target: https://joss.theoj.org/papers/88f3fdfdf621e967da12a9e8fde4785d

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

Welcome to **wristpy**, a Python library for processing and analyzing wrist-worn accelerometer data.

Wristpy provides tools for:

- Loading and calibrating raw accelerometer data
- Computing physical activity levels using several established metrics (ENMO, MAD, Activity- Counts, MIMS)
- Detecting non-wear periods and sleep onset/wake times
- Accessing additional sensor streams (temperature, luminosity, capacitive sense, battery, metadata)

For installation instructions and a quick introduction, see :doc:`getting_started`.

How To Cite
--------------
If you use wristpy in your research, please cite the following paper:

Santorelli, A., Perez, F., de Wael, R. V., Rupprecht, F., d'Antonio-Bertagnolli, J. V., Franco, A., & Kiar, G. (2025).
"wristpy: Fast, User-Friendly Python Processing of Wrist-worn Accelerometer Data", Journal of Open Source Software, 10(114), 8637. https://doi.org/10.21105/joss.08637

Reference
---------

- **GitHub Repository**: `childmindresearch/wristpy <https://github.com/childmindresearch/wristpy>`_
- **Issues & Bug Reports**: `GitHub Issues <https://github.com/childmindresearch/wristpy/issues>`_
- **Contributing**: See :doc:`development`

.. toctree::
   :caption: User Guide
   :maxdepth: 2
   :hidden:

   getting_started.ipynb
   cli_tutorial.ipynb
   python_tutorial.ipynb
   docker_tutorial.ipynb

.. toctree::
   :caption: API Reference
   :maxdepth: 2
   :hidden:

   api/wristpy.core
   api/wristpy.io
   api/wristpy.processing

.. toctree::
   :caption: Development
   :maxdepth: 1
   :hidden:

   development
