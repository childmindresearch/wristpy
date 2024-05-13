"""Calculate base metrics, anglez and enmo."""

import numpy as np
from numpy.lib import stride_tricks

from wristpy.core import models


def euclidean_norm_minus_one(acceleration: models.Measurement) -> models.Measurement:
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

    return models.Measurement(measurements=enmo, time=acceleration.time)


def angle_relative_to_horizontal(
    acceleration: models.Measurement,
) -> models.Measurement:
    """Calculate the angle of the acceleration vector relative to the horizontal plane.

    Args:
        acceleration: the three dimensional accelerometer data. A Measurement object,
        it will have two attributes. 1) measurements, containing the three dimensional
        accelerometer data in an np.array and 2) time, a pl.Series containing
        datetime.datetime objects.

    Returns:
        A Measurement instance containing the values of the angle relative to the
        horizontal plane and the associated timestamps taken from the input unaltered.
        The angle is measured in degrees.
    """
    acceleration_data = acceleration.measurements

    xy_projection_magnitute = np.sqrt(
        (acceleration_data[:, 0] ** 2) + (acceleration_data[:, 1] ** 2)
    )
    angle_relative_to_horizontal_radians = np.arctan(
        acceleration_data[:, 2] / xy_projection_magnitute
    )
    angle_relative_to_horizontal_degrees = np.degrees(
        angle_relative_to_horizontal_radians
    )

    return models.Measurement(
        measurements=angle_relative_to_horizontal_degrees, time=acceleration.time
    )


def rolling_median(
    acceleration: models.Measurement, window_size: int = 51
) -> models.Measurement:
    """Applies rolling mean to acceleration data.

    Args:
        acceleration: _
        window_size: default is 51,

    Returns:
        accel_rolling_mean = the Measurement object with rolling mean applied.
    """
    radius = window_size // 2
    raw_data = acceleration.measurements

    padded_array = np.pad(
        raw_data, ((radius, radius), (0, 0)), mode="constant", constant_values=np.nan
    )

    transposed_array = padded_array.T

    windowed_data = stride_tricks.sliding_window_view(
        transposed_array, window_shape=(window_size,), axis=1
    )
    result = np.nanmedian(windowed_data, axis=-1)

    return models.Measurement(measurements=result.T, time=acceleration.time)
