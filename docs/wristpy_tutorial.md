Wristpy Tutorial
================

Introduction
------------

Wristpy is a Python library designed for processing and analyzing wrist-worn accelerometer data. 
This tutorial will guide you through the basic steps of using Wristpy to analyze your accelerometer data. Specifically,
we will cover the following topics through a few examples:
   - running the default processor, analyzing the output data, and visualizing the results
   - loading data and plotting the raw signals
   - how to calibrate the data, computing ENMO and angle-z from the calibrated data and then plotting those metrics
   - how to obtain non-wear windows and visualize them
   - how to obtain sleep windows and visualize them
      - how we can filter sleep windows that overlap with non-wear


Example 1: Running the default processor
----------------------------------------

The `orchestrator` module of wristpy contains the default processor that will run the entire wristpy processing pipeline. This can be called as simply as:

```python
from wristpy.core import orchestrator

results = orchestrator.run(
   input = '/path/to/your/file.gt3x',
   output = 'path/to/save/file_name.csv'
)
```
This runs the processing pipeline with all the default arguments, creates an output `.csv` file, and will create a `results` object that contains the various output metrics (namely, enmo, angle-z, physical activity values, non-wear detection, sleep detection).

We can visualize some of the outputs within the `results` object, directly, with the following scripts:

Plot the ENMO across the entire data set:
```python
from matplotlib import pyplot as plt
plt.plot(results.enmo.time, results.enmo.measurements)
```

![Example of the ENMO result](enmo_example1.png)

Plot the sleep windows with normalized angle-z data:
```python
from matplotlib import pyplot as plt
plt.plot(results.anglez.time, results.anglez.measurements/90)
plt.plot(results.sleep_windows_epoch.time, results.sleep_windows_epoch.measurements)
plt.legend(['Angle Z', 'Sleep Windows'])
plt.show()
```
![Example of the Sleep and Anglez](sleep_anglez_example1.png)

We can also view and process these outputs from the saved `.csv` output file:

```python
import polars as pl
import matplotlib.pyplot as plt
output_results = pl.read_csv('path/to/save/file_name.csv', try_parse_dates=True)


plt.plot(output_results['time'], output_results['physical_activity_levels'])
```
![Example of plotting physical activity levels from csv](phys_levels_example1.png)

It is also possible to do some analysis on these output variables, for example, if we ant to find the percent of time spent inactive, or in light, moderate, or vigorous physical activity:

```python
inactivity_count = sum(output_results['physical_activity_levels'] == 0)
light_activity_count = sum(output_results['physical_activity_levels'] == 1)
moderate_activity_count = sum(output_results['physical_activity_levels'] == 2)
vigorous_activity_count = sum(output_results['physical_activity_levels'] == 3)
total_activity_count = len(output_results['physical_activity_levels'])

print(f'Light activity percent: {light_activity_count*100/total_activity_count}')
print(f'Moderate activity percent: {moderate_activity_count*100/total_activity_count}')
print(f'Vigorous activity percent: {vigorous_activity_count*100/total_activity_count}')
print(f'Inactivity percent: {inactivity_count*100/total_activity_count}')
```

```
Light activity percent: 12.394840157038699
Moderate activity percent: 1.1030099083940923
Vigorous activity percent: 0.031158471988533682
Inactivity percent: 86.47099146257868
```


Example 2: Loading data and plotting the raw signals
----------------------------------------------------

In this example we will go over the built-in functions to directly read the raw accelerometer and light data, and how to quickly visualize this information.

The built in `readers` module can be used to load all the sensor and metadata from one of the support wrist-watches (`.gt3x` or `.bin`), the reader will automatically select the appropirate loading methodology. 

```python
from wristpy.io.readers import readers

watch_data = readers.read_watch_data('/path/to/geneactive/file.bin')
```

We can then visualize the raw accelerometer and light sensor values very easily as follows:

Plot the raw acceleration along the *x*-axis:

`plt.plot(watch_data.acceleration.time, watch_data.acceleration.measurements[0:1])`

![Plot raw acceleration data from watch_data](raw_accel_example2.png)

Plot the light data:

`plt.plot(watch_data.lux.time, watch_data.lux.measurements)`

![Plot the light data](light_example2.png)



Example 3:  Plot the epoch1 level measurements
----------------------------------------------------
In this example we will expand on the skills learned in `Example2`: we will load the sensor data, calibrate, and then calculate the ENMO and angle-z data in 5s windows (epoch 1 data).

```python
from wristpy.io.readers import readers
from wristpy.processing import calibration, metrics
from wristpy.core import computations

watch_data = readers.read_watch_data('/path/to/geneactive/file.bin')
calibrator_object = calibration.ConstrainedMinimizationCalibration()
calibrated_data = calibrator_object.run_calibration(watch_data.acceleration)

enmo = metrics.euclidean_norm_minus_one(calibrated_data)
anglez = metrics.angle_relative_to_horizontal(calibrated_data)

enmo_epoch1 = computations.moving_mean(enmo)
anglez_epoch1 = computations.moving_mean(anglez)
```

We can then visualize the `epoch1` measurements as:
```python

fig, ax1 = plt.subplots()


ax1.plot(enmo_epoch1.time, enmo_epoch1.measurements, color='blue')
ax1.set_ylabel('ENMO', color='blue')

ax2 = ax1.twinx()
ax2.plot(anglez_epoch1.time, anglez_epoch1.measurements, color='red')
ax2.set_ylabel('Anglez', color='red')

plt.show()
```
![Plot the epoch1 data](enmo_anglez_example3.png)


Example 4: Visualize the detected non-wear times
----------------------------------------------------
In this example we will build on `Example3` by also solving for the non-wear periods, as follows:

```python
from wristpy.io.readers import readers
from wristpy.processing import calibration, metrics


watch_data = readers.read_watch_data('/path/to/geneactive/file.bin')
calibrator_object = calibration.ConstrainedMinimizationCalibration()
calibrated_data = calibrator_object.run_calibration(watch_data.acceleration)
non_wear_array = metrics.detect_nonwear(calibrated_data)

```

We can then visualize the non-wear periods, in comparison to movement (ENMO at the epoch1 level):
```python
from wristpy.core import computations

enmo = metrics.euclidean_norm_minus_one(calibrated_data)
anglez = metrics.angle_relative_to_horizontal(calibrated_data)

enmo_epoch1 = computations.moving_mean(enmo)
anglez_epoch1 = computations.moving_mean(anglez)

plt.plot(enmo_epoch1.time, enmo_epoch1.measurements)
plt.plot(non_wear_array.time, non_wear_array.measurements)

plt.legend(['ENMO Epoch1', 'Non-wear'])
```
![Plot the nonwear periods compared to the ENMO data](nonwear_example4.png)



Example 5: Find and filter the sleep windows
----------------------------------------------------
The following script will obtain the sleep window pairs (onset,wakeup):
  
```python
from wristpy.io.readers import readers
from wristpy.processing import analytics, calibration, metrics


watch_data = readers.read_watch_data('/path/to/geneactive/file.bin')
calibrator_object = calibration.ConstrainedMinimizationCalibration()
calibrated_data = calibrator_object.run_calibration(watch_data.acceleration)
enmo = metrics.euclidean_norm_minus_one(calibrated_data)
anglez = metrics.angle_relative_to_horizontal(calibrated_data)
non_wear_array = metrics.detect_nonwear(calibrated_data)
sleep_detector = analytics.GgirSleepDetection(anglez)
sleep_windows = sleep_detector.run_sleep_detection()
```

Visualize in comparison to the anglez data and the non-wear periods, where sleep periods are visualized by a horizontal blue line, and non-wear periods are visualized with a:

```python
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

# Plot each sleep window as a horizontal line
for sw in sleep_windows:
    if sw.onset is not None and sw.wakeup is not None:
        plt.hlines(1, sw.onset, sw.wakeup, colors='blue', linestyles='solid')

plt.plot(non_wear_array.time, non_wear_array.measurements, color='green')
ax2 = ax1.twinx()
ax2.plot(anglez.time, anglez.measurements, color='red', alpha=0.5)
ax2.set_ylabel('Anglez Epoch1', color='red')

ax1.set_ylabel('Sleep Period/Non-wear')
ax1.set_ylim(0, 1.5)

plt.show()
```
![Plot the sleep periods compared to angelz data.](sleep_angz_example5.png)


The filtered sleep windows can easily be obtained with the following function. This removes all sleep periods that have any overlap with non-wear time.

`filtered_sleep_windows = analytics.remove_nonwear_from_sleep(non_wear_array, sleep_windows )`

And these can be visualized and compared to angle-z and the non-wear periods as previously:
```python
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

for sw in filtered_sleep_windows:
    if sw.onset is not None and sw.wakeup is not None:
        plt.hlines(1, sw.onset, sw.wakeup, colors='blue', linestyles='solid')

plt.plot(non_wear_array.time, non_wear_array.measurements, color='green')
ax2 = ax1.twinx()
ax2.plot(anglez.time, anglez.measurements, color='red', alpha=0.5)
ax2.set_ylabel('Anglez Epoch1', color='red')

ax1.set_ylabel('Sleep Period/Non-wear')
ax1.set_ylim(0, 1.5)

plt.show()
```
![Plot the filtered sleep windows.](filtered_sleep_windows.png)