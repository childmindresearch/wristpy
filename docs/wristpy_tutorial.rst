Wristpy Tutorial
================

Introduction
------------

Wristpy is a Python library designed for processing and analyzing wrist-worn accelerometer data. 
This tutorial will guide you through the basic steps of using Wristpy to analyze your accelerometer data.

Installation
------------

You can install Wristpy using pip:

.. code-block:: bash

   pip install wristpy


Components of wristpy
---------------------
- load data
   - access to metadata
- calibrate data
- calculate metrics

Basic Usage
-----------

First, import the necessary modules:

.. code-block:: python

   from wristpy import orchestrator, models

Next, load your data. In this example, we'll use a sample data file:

.. code-block:: python

   data_file = 'path_to_your_data_file'

Now, you can run the orchestrator on your data:

.. code-block:: python

   results = orchestrator.run(input=data_file)

The `run` function returns a `Results` object that contains the analysis results. You can access the results like this:

.. code-block:: python

   enmo = results.enmo
   anglez = results.anglez
   # etc.

Each result is a `Measurement` object that contains the measurements for that metric.

Some examples
----------

Example 1
- load data
plot raw acceleration

Example 2
- load data
- calibrate data
-compute metrics
plot enmo and anglez

Example 3
 - get nonwear, plot vs enmo

Example 4
 - get sleep, plot vs anglez
 - plot sleep, nonwear, and anglez
 - filter nonwear from sleep, plot filtered sleep vs anglez vs nonwear