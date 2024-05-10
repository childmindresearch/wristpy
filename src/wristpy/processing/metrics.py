"""Calculate base metrics, anglez and enmo."""

import numpy as np

from wristpy.core.models import Measurement


def euclidean_norm_minus_one(acceleration: Measurement) -> Measurement:
    """Compute ENMO, the Euclidean Norm Minus One (1 standard gravity unit).

    Negative values of ENMO are set to zero because ENMO is meant as a measure of
    physival activity. Negative values steming from imperfect calibration or noise from
    the device would have no meaningful interpretation in this context and would be
    detrimental to the intended analysis.

    Args:
        acceleration: the three dimensional accelerometer data. A Measurement object,
        it will have two attributes. 1) measurements, containing the three dimensional
        accelerometer data in an np.array and 2) time, a pl.Series containing
        datetime.datetime objects.

    Returns:
        A Measurement object containing the calculated ENMO values and the
        associated time stamps taken from the input.
    """
    enmo = np.linalg.norm(acceleration.measurements, axis=1) - 1

    enmo = np.maximum(enmo, 0)

    return Measurement(measurements=enmo, time=acceleration.time)


def angle_relative_to_horizontal(acceleration: Measurement) -> Measurement:
    """Calculate the angle of the acceleration vector relative to the horizontal plane.

    Args:
        acceleration: the three dimensional accelerometer data. A Measurement object,
        it will have two attributes. 1) measurements, containing the three dimensional
        accelerometer data in an np.array and 2) time, a pl.Series containing
        datetime.datetime objects.

    Returns:
        A Measurement instance containing the values of the angle relative to the
        horizontal plane and the associated timestamps taken from the input unaltered.
    """
    acceleration_data = acceleration.measurements

    xy_projection_magnitute = np.sqrt(
        (acceleration_data[:, 0] ** 2) + (acceleration_data[:, 1] ** 2)
    )
    angle_z_radians = np.arctan(acceleration_data[:, 2] / xy_projection_magnitute)
    angle_z_degrees = np.degrees(angle_z_radians)

    return Measurement(measurements=angle_z_degrees, time=acceleration.time)
