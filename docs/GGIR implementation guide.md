**GGIR implementation guide:**



**Phase 1: Calibration, calculate ENMO and anglez**

- [Calibration:](#calibration)
  - [Summary of process:](#summary-of-process)
    - [**References**](#references)
- [Calcuate metrics](#calcuate-metrics)
  - [Queries](#queries)
- [**Non-wear time**](#non-wear-time)
    - [**References**](#references-1)


# Calibration:
We expect that for non-movement time the applied g-force acting on the sensor should be 1g. Find non-movement time, compare data points during this time to unit sphere. Solve a fitting problem (apply scale and offset to data) to get those points as close as possible to unit sphere. Then apply this scale and offset to all data:

$$
\tilde{s_i}(t) = a_i * s_i(t) + b_i
$$

Where, $s_i$ is the initial accelerometer data for each i-th dimension, $a_i$ is the scaling factor, $b_i$ is the offset factor, and $\tilde{s_i}$ is the calibrated accelerometer signal.

## Summary of process:
1. Find non-movement times: 
   - Defined as when SD(acc[x,y,z])<0.013 over 10s window
   - This is done by downsampling acc[x,y,z]. 
   - Then SD over each window. 
   - Identify windows where SD, in all 3 axes, is < cutoff_value (default 0.013). Keep only those windows. Also, “& npall( abs(roll_mean(acc))<2)”, to ‘avoid clipping’ (removes sections with a large mean but no SD).

2. Check for sufficient data points + "sphere well populated"
   - Defined by GGIR a minimum of 12 hours of data. Breaks data into 12 hour chunks and tries to calibrate over chunks. If first chunk is unsuccessful, moves to next chunk. If successful in the first chunk, ends, and applies calibration. (NOTE, is this the right way to do this??)
   - "To minimize signal processing time, the autocalibration methodinitially only uses the first 72 h (3 days) of a measurement file basedon which calibration error reduction is evaluated. If the file length is 3 days, then all available data are used. If calibration error is notreduced to 10 mgor if the 300-mgcriteria for ellipsoid datasparseness is not met, additional chunks of 12-h data are iterativelyadded until either error and sparseness criteria are met or until the endof the file is reached."
  
   - Sphere "well and sparsely populated", *spherecrit*, checks that there are points above a predetermined cutoff value *g* force (default 0.3), in both positive and negative directions (mean(accel) < -*spherecrit*) && (mean(accel) > *spherecrit*)
3. 	Find calibration error from only those windows (trim accel[x,y,z] to 10s  windows that satisfy above criteria):
  - Error defined as $err(t)= \sqrt{x(t)^2+y(t)^2+z(t)^2 }-1$; *norm(mean(acc)) -1*
  - Iterative closest point fit until *max_iter* or *err<tol_er*
  - Solve for the three scaling and offset values
  - Check if new calibration error is < cal_error_start && 0.01 (apply offset + scale to windowed acc(x,y,z));    (example from scikit)     
```python
        acc_rm = (acc_rm + offset) * scale + tmp_rm * tmp_scale
        cal_error_end = around(mean(abs(norm(acc_rm, axis=1) - 1)), decimals=5)
        assess if calibration error has been significantly improved
                if (cal_error_end < cal_error_start) and (cal_error_end < 0.01):
                    return True, offset, scale, tmp_scale, tmp_mean
                else:
                    return False, offset, scale, tmp_scale, tmp_mean
```
  - Can include temperature data too (FUTURE)
  - If successful calibration, apply calibration to raw accelerometer data. Return calibrated data, as well as scale, offset, and final cal_error.

### **References**
Source code : https://rdrr.io/cran/GGIR/src/R/g.calibrate.R
Python implementation from Pfizer : https://github.com/pfizer-opensource/scikit-digital-health/blob/main/src/skdh/preprocessing/calibrate.py
Paper:
https://journals.physiology.org/doi/epdf/10.1152/japplphysiol.00421.2014


# Calcuate metrics

Key physical activity metrics are calculated from the calibrated, raw accelerometer data. They are then downsampled (mean over the window length) over the three “epochs”. GGIR uses three window lengths, with some rules about length of each: shortest must be at least 1 second, the largest window must be a multiple of the second.  Key metrics we will focus on in Phase 1, are ENMO (Euclidean norm, minus one) and anglez (angle relative to z-axis). These are calculated at the sample level and then a moving mean over the desired window lengths is applied. These metrics are used for all subsequent physical activity measures and sleep detection (in GGIR).

## Queries
Key note about ENMO: default in GGIR and scikit is to trim any negative values to 0. Scikit has option to keep absolute values. Some questions about these metrics, mostly for the future:

- Should we include abs(ENMO)?
- Other desirable metrics, there is a long list included in GGIR, including filtering methods? are these used in literature?
- GGIR calculates the anglez metric using a rolling median of acc(x,y,z). This rolling median is calculated over a window that is defined as: 5* sampling_rate +1, unless sampling_rate is > 10, then it is hardcoded to 5*10 +1. Scikit does not do this (and in my preliminary test I did not either and it didn’t seem to cause any issues). Not sure the need for this, especially since angle_z is downsampled after being calculated.


# **Non-wear time**

From the scikit paper for the DETACH algorithm

"Final algorithm rules
Based on constraints of the CART approach and observations made via edge cases, the outputs of the decisions tree classifiers were supplemented to produce a more intuitive and robust algorithm that was not specific to GENEActiv devices [37]. Specifically, conditions of the decision tree analysis required that each axis of accelerometry be considered separately within both non-wear start and non-wear end. A recent paper by [26] however, demonstrated that using two or three axes within the parameter set optimizes non-wear detection. As such, the DETACH algorithm was refined to include acceleration features for multiple, non-specific, axes for both start and end of non-wear. Given the number of axes considered within the final set of rules, the threshold for SD of acceleration was increased slightly from 7 to 8 mg. Specifically, for non-wear start, SD of acceleration over the following minute for 2 or more axes was set to be < 8 mg. This was combined with the need for the SD of acceleration of 2 or more axes to be 8 mg for at least 90% of the following 5 minutes. The 90% requirement was intended to avoid cases where spurious movement occurring during non-wear (e.g., a device being bumped) would be misinterpreted as wear. Additionally, the rate-of-change in temperature criterion for non-wear start was modified slightly from − 0.27 to − 0.2 but the absolute temperature threshold remained < 30 °C (rounded). To detect a non-wear end event, SD of acceleration during the previous minute was set to be > 8 mg for all axes, combined with a SD of acceleration > 8 mg for two or more axes for at least 50% of the following 5-minute window. This SD condition for non-wear end is followed by the need for the 5-minute temperature rate-of-change to be > 0.1 °C/minute.

Visual inspection of non-wear periods in the test dataset that were not detected using the above pathways revealed cases in which the ambient temperature was close to the near-body temperature at the start of a non-wear period or rate-of-change didn’t meet the threshold of a minimum 0.2 °C decrease per minute. To address these possibilities, additional non-wear criteria focusing on absolute temperature were created to accompany the rate-of-change criteria. While maintaining the rules for accelerometry, the additional criteria serve to identify periods where temperature is below 26 °C for the start of a non-wear period or above 26 °C for non-wear end. Table 1 summarizes the non-wear rules based on decision tree classifiers, published literature, and edge case observations (algorithm available, see data availability statement)."

### **References**
GGIR source : https://rdrr.io/cran/GGIR/src/R/g.applymetrics.R

Python Scikit implementation : https://github.com/pfizer-opensource/scikit-digital-health/blob/main/src/skdh/activity/metrics.py
