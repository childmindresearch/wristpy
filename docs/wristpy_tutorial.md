Wristpy Tutorial
================

Introduction
------------

Wristpy is a Python library designed for processing and analyzing wrist-worn accelerometer data. 
This tutorial will guide you through the basic steps of using Wristpy to analyze your accelerometer data. Specifically,
we will cover the following topics:
   - running the default processor, analyzing the output data and visualizing the results
   - loading data and plotting the raw signals
   - how to calibrate the data, and the different options, computing ENMO and angle-z from the calibrated data and then plotting those metrics
   - how to obtain non-wear windows and visualize them
   - how to obtain sleep windows and visualize them
      - how we can filter sleep windows that overlap with non-wear


Example 1: Running the default processor
----------------------------------------



Example 2: Loading data and plotting the raw signals
----------------------------------------------------
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