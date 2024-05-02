"""Calculate base metrics, anglez and enmo."""

import numpy as np

from wristpy.core.models import Measurement


def euclidean_norm(acceleration: Measurement)-> Measurement:
    """Compute ENMO, the Euclidean Norm Minus One (1 standard gravity unit).

    Args:
        acceleration: the three dimensional accelerometer data. A Measurement object,
        it will have two attributes. 1) measurements, containing the three dimensional 
        accelerometer data in an np.array and 2) time, a pl.Series containing 
        datetime.datetime objects.

    Returns:
        A Measurement object containing the calculated ENMO values and the 
        associated time stamps taken from the input.
    """
    #compute euclidean norm - 1
    enmo = np.linalg.norm(acceleration.measurements, axis = 1) - 1

    #GGIR replaces negative values with 0
    enmo = np.maximum(enmo, 0)

    return Measurement(measurements= enmo, time = acceleration.time)