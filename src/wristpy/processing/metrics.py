"""Calculate base metrics, anglez and enmo."""

from typing import Literal, Tuple

from typing import Literal, Tuple

import numpy as np
import polars as pl
from scipy import interpolate, signal
from skdh.preprocessing import wear_detection

from wristpy.core import computations, config, models
from wristpy.io.readers import readers
from wristpy.processing import mims

logger = config.get_logger()


def euclidean_norm_minus_one(
    acceleration: models.Measurement, epoch_length: float = 5.0
) -> models.Measurement:
    """Compute ENMO, the Euclidean Norm Minus One (1 standard gravity unit).

    Negative values of ENMO are set to zero because ENMO is meant as a measure of
    physival activity. Negative values steming from imperfect calibration or noise from
    the device would have no meaningful interpretation in this context and would be
    detrimental to the intended analysis.

    The data can be downsampled to a temporal resolution of `epoch_length` seconds.

    Args:
        acceleration: the three-dimensional accelerometer data. A Measurement object,
        it will have two attributes. 1) measurements, containing the three-dimensional
        accelerometer data in an np.array and 2) time, a pl.Series containing
        datetime.datetime objects.
        epoch_length: The temporal resolution of the downsampled enmo, in seconds.
            Must be greater than 0. Defaults to 5.0s.

    Returns:
        A Measurement object containing the calculated ENMO values and the
        associated time stamps taken from the input.
    """
    enmo = np.linalg.norm(acceleration.measurements, axis=1) - 1

    enmo = np.maximum(enmo, 0)

    return computations.moving_mean(
        models.Measurement(measurements=enmo, time=acceleration.time), epoch_length
    )


def angle_relative_to_horizontal(
    acceleration: models.Measurement, epoch_length: float = 5.0
) -> models.Measurement:
    """Calculate the angle of the acceleration vector relative to the horizontal plane.

     The data can be downsampled to a temporal resolution of `epoch_length` seconds.

    Args:
        acceleration: the three-dimensional accelerometer data. A Measurement object,
        it will have two attributes. 1) measurements, containing the three-dimensional
        accelerometer data in a np.array and 2) time, a pl.Series containing
        datetime.datetime objects.
        epoch_length: The temporal resolution of the downsampled anglez, in seconds.
            Must be greater than 0. Defaults to 5.0s.

    Returns:
        A Measurement instance containing the values of the angle relative to the
        horizontal plane and the associated timestamps taken from the input unaltered.
        The angle is measured in degrees.
    """
    xy_projection_magnitute = np.linalg.norm(acceleration.measurements[:, 0:2], axis=1)

    angle_radians = np.arctan(acceleration.measurements[:, 2] / xy_projection_magnitute)

    angle_degrees = np.degrees(angle_radians)

    return computations.moving_mean(
        models.Measurement(measurements=angle_degrees, time=acceleration.time),
        epoch_length,
    )


def mean_amplitude_deviation(
    acceleration: models.Measurement, epoch_length: float = 5.0
) -> models.Measurement:
    """Calculate the mean amplitude deviation of the acceleration data.

    An alternative to ENMO to quantify the intensity of physical activity.
    It is calculated as the mean of the absolute difference between acceleration values
    and the mean of those acceleration values in a 5s window.

    Args:
        acceleration: the calibrated acceleration data.
        epoch_length: The length of the window in seconds.

    Returns:
        A new Measurement object containing the MAD values.

    References:
        Vähä-Ypyä H, Vasankari T, Husu P, Suni J, Sievänen H. A universal, accurate
        intensity-based classification of different physical activities using raw data
        of accelerometer. Clin Physiol Funct Imaging. 2015 Jan;35(1):64-70.
        doi: 10.1111/cpf.12127. Epub 2014 Jan 7. PMID: 24393233.
    """
    acceleration_magnitude = np.linalg.norm(acceleration.measurements, axis=1)

    mad_lf = pl.LazyFrame(
        {"time": acceleration.time, "acceleration_magnitude": acceleration_magnitude}
    ).set_sorted("time")

    mad_df = (
        mad_lf.group_by_dynamic(index_column="time", every=f"{int(epoch_length*1e9)}ns")
        .agg(
            [
                (
                    pl.col("acceleration_magnitude")
                    - pl.col("acceleration_magnitude").mean()
                )
                .abs()
                .mean()
                .alias("mean_amplitude_deviation")
            ]
        )
        .collect()
    )

    return models.Measurement.from_data_frame(mad_df)


def actigraph_activity_counts(
    acceleration: models.Measurement, epoch_length: float = 5.0
) -> models.Measurement:
    """Compute Actigraph acitivty counts.

    This function computes the Actigraph activity counts based on [1].
    The acceleartion data is downsample to 30Hz, bandpass filtered, scaled,
    and then thresholded. The counts are then summed along each axis for
    the chosen epoch length.

    Args:
        acceleration: The calibrated acceleration data.
        epoch_length: The length of the epoch in seconds, defaults to 60s.

    Returns:
        The activity counts as a Measurement object.

    References:
        [1] A. Neishabouri et al., “Quantification of acceleration as activity counts
        in ActiGraph wearable,” Sci Rep, vol. 12, no. 1, Art. no. 1, Jul. 2022,
        doi: 10.1038/s41598-022-16003-x.
    """
    logger.debug("Running activty count physical activity metric.")

    epoch_length_nanoseconds = round(epoch_length * 1e9)

    # input and output coefficients for the bandpass filter from [1]
    input_coef = np.array(
        [
            -0.009341062898525,
            -0.025470289659360,
            -0.004235264826105,
            0.044152415456420,
            0.036493718347760,
            -0.011893961934740,
            -0.022917390623150,
            -0.006788163862310,
            0.000000000000000,
        ],
        dtype=np.float64,
    )
    output_coef = np.array(
        [
            1.00000000000000000000,
            -3.63367395910957000000,
            5.03689812757486000000,
            -3.09612247819666000000,
            0.50620507633883000000,
            0.32421701566682000000,
            -0.15685485875559000000,
            0.01949130205890000000,
            0.00000000000000000000,
        ],
        dtype=np.float64,
    )

    # scaling factor from [1]
    scaling_factor = (3 / 4096) / (2.6 / 256) * 237.5

    acceleration_30hz = computations.resample(acceleration, 1 / 30)

    initial_conditions = signal.lfilter_zi(input_coef, output_coef).reshape((-1, 1))

    acceleration_bpf, _ = signal.lfilter(
        input_coef,
        output_coef,
        acceleration_30hz.measurements,
        zi=np.repeat(
            initial_conditions, acceleration_30hz.measurements.shape[1], axis=-1
        )
        * acceleration_30hz.measurements[0],
        axis=0,
    )

    scaled_acceleration = acceleration_bpf * scaling_factor
    threshold_acceleration = np.floor(np.minimum(np.abs(scaled_acceleration), 128))
    threshold_acceleration[threshold_acceleration < 4] = 0

    acceleration_10hz = computations.resample(
        models.Measurement(
            measurements=threshold_acceleration, time=acceleration_30hz.time
        ),
        1 / 10,
    )
    acceleration_10hz.measurements = np.floor(acceleration_10hz.measurements)

    aggregator = pl.exclude(["time"]).drop_nans()
    epoch_counts = (
        acceleration_10hz.lazy_frame()
        .group_by_dynamic("time", every=f"{epoch_length_nanoseconds}ns")
        .agg(aggregator.sum())
    ).collect()

    ag_counts = epoch_counts.with_columns(
        (pl.col("column_0") ** 2 + pl.col("column_1") ** 2 + pl.col("column_2") ** 2)
        .sqrt()
        .alias("magnitude")
    )
    ag_counts = ag_counts.drop(["column_0", "column_1", "column_2"])

    return models.Measurement.from_data_frame(ag_counts)


def detect_nonwear(
    acceleration: models.Measurement,
    short_epoch_length: int = 900,
    n_short_epoch_in_long_epoch: int = 4,
    std_criteria: float = 0.013,
) -> models.Measurement:
    """Set non_wear_flag based on accelerometer data.

    This implements a modified version of the GGIR "2023" non-wear detection algorithm.
    Briefly, the algorithm, creates a sliding window of long epoch length that steps
    forward by the short epoch length. The long epoch length is an integer multiple of
    the short epoch length, that can be specified by the user.
    It checks if the acceleration data in that long window, for each axis, meets the
    criteria threshold for the standard deviation of acceleration values to
    compute a non-wear value. The total non-wear value (0, 1, 2, 3) for the long window
    is the sum of each axis.
    The non-wear value is applied to all the short windows that make up the long
    window. Additionally, as the majority of the short windows are part of multiple long
    windows, the value of a short window is updated to the maximum nonwear value from
    these overlaps.
    Finally, there is a pass to find isolated "1s" in the non-wear
    value, and set them to 2 if surrounded by > 1 values. The non-wear flag is set to
    1 (true) if the non-wear value is >= 2, and 0 (false) otherwise.


    Args:
        acceleration: The Measurment instance that contains the calibrated acceleration
            data.
        short_epoch_length: The short window size, in seconds.
        n_short_epoch_in_long_epoch: Number of short epochs that makeup one long epoch.
        std_criteria: Threshold criteria for standard deviation.


    Returns:
        A new Measurment instance with the non-wear flag and corresponding timestamps.
    """
    logger.debug("Detecting non-wear data.")
    acceleration_grouped_by_short_window = _group_acceleration_data_by_time(
        acceleration, short_epoch_length
    )

    nonwear_value_array = _compute_nonwear_value_array(
        acceleration_grouped_by_short_window,
        n_short_epoch_in_long_epoch,
        std_criteria,
    )

    nonwear_value_array_cleaned = _cleanup_isolated_ones_nonwear_value(
        nonwear_value_array
    )
    non_wear_flag = np.where(nonwear_value_array_cleaned >= 2, 1, 0)

    return models.Measurement(
        measurements=non_wear_flag, time=acceleration_grouped_by_short_window["time"]
    )


def combined_temp_accel_detect_nonwear(
    acceleration: models.Measurement,
    temperature: models.Measurement,
    std_criteria: float = 0.013,
    temperature_threshold: float = 26.0,
) -> models.Measurement:
    """This function uses the both temperature and acceleration data to detect non-wear.

    The function implements the algorithm described in the Zhou 2015 paper. Briefly,
    data is chunked into one minute windows. Each windows is classified as "wear" or
    "non-wear", here denoted by 0 and 1, respectively.

    Args:
        acceleration: The Measurment instance that contains the calibrated acceleration.
        temperature: The Measurment instance that contains the temperature data.
        std_criteria: Threshold criteria for standard deviation.
        temperature_threshold: The temperature threshold for non-wear detection.

    Returns:
        A new Measurment instance with the non-wear flag and corresponding timestamps.
        The temporal resolutions is one minute.

    References:
        Zhou S, Hill RA, Morgan K, et alClassification of accelerometer wear and
        non-wear events in seconds for monitoring free-living physical activityBMJ
        Open 2015; 5:e007447. doi: 10.1136/bmjopen-2014-007447
    """
    logger.debug("Combined temperature and accelereation non-wear detection algorithm.")

    acceleration_grouped_by_window = _group_acceleration_data_by_time(acceleration, 60)
    low_pass_temp = _pre_process_temperature(temperature, acceleration)

    temperature_grouped_by_window = low_pass_temp.group_by_dynamic(
        index_column="time", every="60s"
    ).agg([pl.exclude(["time"])])

    total_n_short_windows = len(acceleration_grouped_by_window)

    nonwear_value_array = np.zeros(total_n_short_windows)

    for window_n in range(total_n_short_windows):
        mean_temp = temperature_grouped_by_window["temperature"][window_n].mean()
        axis_nonwear_value = (
            acceleration_grouped_by_window[window_n]
            .select(
                pl.col("X", "Y", "Z").map_batches(
                    lambda df: _compute_nonwear_value_per_axis(
                        df,
                        std_criteria,
                    )
                )
            )
            .sum_horizontal()
            .to_numpy()
        )
        if mean_temp < temperature_threshold and (axis_nonwear_value >= 2):
            nonwear_value_array[window_n] = 1
        elif window_n > 0 and mean_temp < temperature_threshold:
            mean_temp_previous = temperature_grouped_by_window["temperature"][
                window_n - 1
            ].mean()
            if mean_temp <= mean_temp_previous:
                nonwear_value_array[window_n] = 1

    return models.Measurement(
        measurements=nonwear_value_array,
        time=acceleration_grouped_by_window["time"],
    )


def detach_nonwear(
    acceleration: models.Measurement,
    temperature: models.Measurement,
    std_criteria: float = 0.013,
) -> models.Measurement:
    """This function implements the scikit DETACH algorithm for non-wear detection.

    The nonwear array is downsampled to 60 second resolution.

    Args:
        acceleration: The Measurment instance that contains the calibrated acceleration.
        temperature: The Measurment instance that contains the temperature data.
        std_criteria: The acceleration STD threshold for non-wear detection.

    Returns:
        A new Measurment instance with the non-wear flag and corresponding timestamps.
        The temporal resolutions is one minute.

    References:
        A. Vert et al., “Detecting accelerometer non-wear periods using change
        in acceleration combined with rate-of-change in temperature,” BMC Medical
        Research Methodology, vol. 22, no. 1, p. 147, May 2022,
        doi: 10.1186/s12874-022-01633-6.
    """

    def upsample_temperature(
        temperature: np.ndarray, acceleration: np.ndarray
    ) -> np.ndarray:
        """Helper function to upsample the temperature to match acceleration data."""
        x_temperature = np.linspace(0, 1, len(temperature))
        x_acceleration = np.linspace(0, 1, len(acceleration))
        temperature_interpolator = interpolate.interp1d(x_temperature, temperature)
        return temperature_interpolator(x_acceleration)

    def cleanup_DETACH_nonwear(nonwear: models.Measurement) -> models.Measurement:
        """Helper function to downsample DETACH ouptut to 60s."""
        nonwear_downsample = computations.moving_mean(nonwear, 60)

        nonwear_downsample.measurements = np.where(
            nonwear_downsample.measurements >= 0.5, 1, 0
        )

        return nonwear_downsample

    logger.debug("Running scikit DETACH algorithm.")
    time = acceleration.time.dt.timestamp(time_unit="ns").to_numpy()
    upsample_temp = upsample_temperature(
        temperature.measurements, acceleration.measurements
    )

    detach_class = wear_detection.DETACH(sd_thresh=std_criteria)
    nonwear_array = np.ones(len(time), dtype=np.float32)
    detach_wear = detach_class.predict(
        time=time / 1e9, accel=acceleration.measurements, temperature=upsample_temp
    )

    for start, stop in detach_wear["wear"]:
        nonwear_array[start:stop] = 0

    non_wear_time = readers.unix_epoch_time_to_polars_datetime(time)

    nonwear = models.Measurement(measurements=nonwear_array, time=non_wear_time)

    return cleanup_DETACH_nonwear(nonwear)


def _group_acceleration_data_by_time(
    acceleration: models.Measurement, window_length: int
) -> pl.DataFrame:
    """Helper function to group the acceleration data by short windows.

    Args:
        acceleration: The Measurment instance that contains the calibrated acceleration.
        window_length: The window size, in seconds.

    Returns:
        A polars DataFrame with the acceleration data grouped by window_length.
    """
    acceleration_polars_df = pl.DataFrame(
        {
            "X": acceleration.measurements[:, 0],
            "Y": acceleration.measurements[:, 1],
            "Z": acceleration.measurements[:, 2],
            "time": acceleration.time,
        }
    )
    acceleration_polars_df = acceleration_polars_df.with_columns(
        pl.col("time").set_sorted()
    )

    acceleration_grouped_by_window_length = acceleration_polars_df.group_by_dynamic(
        index_column="time", every=(str(window_length) + "s")
    ).agg([pl.all().exclude(["time"])])

    return acceleration_grouped_by_window_length


def _compute_nonwear_value_array(
    grouped_acceleration: pl.DataFrame,
    n_short_epoch_in_long_epoch: int,
    std_criteria: float,
) -> np.ndarray:
    """Helper function to calculate the nonwear value array.

    This function calculates the nonwear value array based on the GGIR 2023 methodology.
    It computes the nonwear value for each axis, based on the acceleration data that
    makes up one long epoch window. That nonwear value is then applied to all the
    short windows that make up the long window. It iterates forward by one short_window
    length and repeats the process. For the overlapping short windows, the maximum
    nonwear value is kept and is assigned to the nonwear value array.

    Args:
        grouped_acceleration: The acceleration data grouped into short windows.
        n_short_epoch_in_long_epoch: Number of short epochs that makeup one long epoch.
        std_criteria: Threshold criteria for standard deviation.

    Returns:
        Non-wear value array.
    """
    total_n_short_windows = len(grouped_acceleration)
    nonwear_value_array = np.zeros(total_n_short_windows)

    for window_n in range(total_n_short_windows - n_short_epoch_in_long_epoch + 1):
        acceleration_selected_long_window = grouped_acceleration[
            window_n : window_n + n_short_epoch_in_long_epoch
        ]

        calculated_nonwear_value = acceleration_selected_long_window.select(
            pl.col("X", "Y", "Z").map_batches(
                lambda df: _compute_nonwear_value_per_axis(
                    df,
                    std_criteria,
                )
            )
        ).sum_horizontal()

        max_window_value = np.maximum(
            nonwear_value_array[window_n : window_n + n_short_epoch_in_long_epoch],
            np.repeat(calculated_nonwear_value, n_short_epoch_in_long_epoch),
        )
        nonwear_value_array[window_n : window_n + n_short_epoch_in_long_epoch] = (
            max_window_value
        )

    return nonwear_value_array


def _compute_nonwear_value_per_axis(
    axis_acceleration_data: pl.Series,
    std_criteria: float,
) -> bool:
    """Helper function to calculate the nonwear criteria per axis.

    Args:
        axis_acceleration_data: The long window acceleration data for one axis.
            It is a pl.Series chunked into short windows where each row is a list of the
            acceleration data of one axis (length of each list is the number of samples
            that make up short_epoch_length in seconds).
        std_criteria: Threshold criteria for standard deviation


    Returns:
        Non-wear value for the axis.
    """
    axis_long_window_data = pl.concat(axis_acceleration_data, how="vertical")
    axis_std = axis_long_window_data.std()
    criteria_boolean = axis_std < std_criteria

    return criteria_boolean


def _cleanup_isolated_ones_nonwear_value(nonwear_value_array: np.ndarray) -> np.ndarray:
    """Helper function to clean up isolated ones in nonwear value array.

    This function finds isolated ones in the nonwear value array and
    sets them to 2 if they are surrounded by values > 1.

    Args:
        nonwear_value_array: The nonwear value array that needs to be cleaned up.
            It is a 1D numpy array.

    Returns:
        The modified nonwear value array.
    """
    nonwear_value_array = nonwear_value_array.astype(int)

    left_neighbors = np.roll(nonwear_value_array, 1)
    right_neighbors = np.roll(nonwear_value_array, -1)

    condition = (left_neighbors > 1) & (right_neighbors > 1) & nonwear_value_array == 1
    condition[0] = False
    condition[-1] = False

    nonwear_value_array[condition] = 2

    return nonwear_value_array


def _pre_process_temperature(
    temperature: models.Measurement, acceleration: models.Measurement
) -> pl.DataFrame:
    """Pre-process temperature data for non-wear detection.

    This function is used specificall for the CTA, it ensures that the temperature
    and acceleration Measurements end at the same time (due to the inherent sampling
    rate differences on the device), upsamples the temperature data to 1 second
    intervals and then low-pass filters the data by applying a 2 minute rolling mean.

    Args:
        temperature: The Measurment instance that contains the temperature data.
        acceleration: The Measurment instance that contains the acceleration data.

    Returns:
        A polars DataFrame with the pre-processed temperature data.

    Raises:
        ValueError: If the acceleration data ends before the temperature data.
    """
    temperature_polars_df = pl.DataFrame(
        {
            "time": temperature.time,
            "temperature": temperature.measurements,
        }
    )

    if acceleration.time[-1] > temperature.time[-1]:
        new_row = pl.DataFrame(
            {
                "time": acceleration.time[-1],
                "temperature": temperature.measurements[-1],
            },
            schema={"time": pl.Datetime("ns"), "temperature": pl.Float32},
        )

        temperature_polars_df = temperature_polars_df.vstack(new_row)
    elif temperature.time[-1] > acceleration.time[-1]:
        raise ValueError("Temperature data ends before acceleration data.")

    temperature_polars_df = temperature_polars_df.with_columns(
        pl.col("time").set_sorted()
    )
    upsample_temp = temperature_polars_df.upsample(
        time_column="time", every="1s", maintain_order=True
    ).interpolate()
    upsample_temp = upsample_temp.with_columns(
        pl.col("temperature").fill_null(strategy="forward")
    )

    low_pass_temp = upsample_temp.rolling(index_column="time", period="120s").agg(
        [pl.mean("temperature")]
    )
    return low_pass_temp


def monitor_independent_movement_summary_units(
    acceleration: models.Measurement,
    combination_method: Literal["sum", "vector_magnitude"] = "sum",
    epoch: float = 60.0,
    interpolation_frequency: int = 100,
    dynamic_range: Tuple[float, float] = (-8.0, 8.0),
    cutoffs: Tuple[float, float] = (0.2, 5.0),
    order: int = 4,
    *,
    rectify: bool = True,
) -> models.Measurement:
    """Calculates monitor independent movement summary units (MIMS).

    This function processes raw acceleration data to calculate MIMS units,
    based on the original R implementation by John et al.(2019). The main processing
    steps as described in  the original paper are interpolation (100Hz by default),
    extrapolation of any values that went beyond the dynamic range of the device,
    filtering using a 4th order butterworth filter, aggregation by calculating the area
    under the curve over a given epoch, and finally trunction of small values.
    The MIMS value per axis is then combined through a sum or vector magnitude, and
    returned as a single vector.

    Args:
        acceleration: Triaxial acceleration data to be processed.
        combination_method: Method to combine MIMS values accross axes.
        epoch: Duration over which each MIMS value will be calculated. Measured in
            seconds.
        interpolation_frequency: Frequency to interpolate acceleration data, defaults
            to 100 Hz as described in John et al. 2019.
        dynamic_range: Tuple specifying the minimum and maximum values (in g) of the
            accelerometer's dynamic range.
        cutoffs: Tuple specifying the low and high cutoff frequencies (in Hz) for the
            Butterworth filter.
        order: Order of the Butterworth filter.
        rectify:If True, rectifies the filtered signal before aggregation. ,

    Returns:
        Processed MIMS values after combining the values of each axis with the provided
        method.

    References:
        John, D., Tang, Q., Albinali, F. and Intille, S., 2019. An Open-Source
        Monitor-Independent Movement Summary for Accelerometer Data Processing. Journal
        for the Measurement of Physical Behaviour, 2(4), pp.268-281.

    """
    interpolated_measure = mims.interpolate_measure(
        acceleration=acceleration,
        new_frequency=interpolation_frequency,
    )
    extrapolated_measure = mims.extrapolate_points(
        acceleration=interpolated_measure,
        dynamic_range=dynamic_range,
        sampling_rate=interpolation_frequency,
    )
    filtered_measure = mims.butterworth_filter(
        acceleration=extrapolated_measure,
        sampling_rate=interpolation_frequency,
        cutoffs=cutoffs,
        order=order,
    )
    aggregated_measure = mims.aggregate_mims(
        acceleration=filtered_measure,
        epoch=epoch,
        sampling_rate=interpolation_frequency,
        rectify=rectify,
    )
    aggregated_measure.measurements = np.where(
        aggregated_measure.measurements <= 0.001, 0, aggregated_measure.measurements
    )
    combined_mims = mims.combine_mims(
        acceleration=aggregated_measure, combination_method=combination_method
    )
    return combined_mims
