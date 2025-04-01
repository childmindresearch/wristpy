"""Python based runner."""

import datetime
import itertools
import logging
import pathlib
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

from wristpy.core import computations, config, exceptions, models
from wristpy.io.readers import readers
from wristpy.processing import (
    analytics,
    calibration,
    idle_sleep_mode_imputation,
    metrics,
    nonwear_utils,
)

logger = config.get_logger()

VALID_FILE_TYPES = (".csv", ".parquet")


def format_sleep_data(
    sleep_windows: List[analytics.SleepWindow], reference_measure: models.Measurement
) -> np.ndarray:
    """Formats sleep windows into an array for saving.

    Args:
        sleep_windows: The list of time stamp pairs indicating periods of sleep.
        reference_measure: The measure from which the temporal resolution will be taken.

    Returns:
        1-D np.ndarray, with 1 indicating sleep. Will be of the same length as
            the timestamps in the reference_measure.
    """
    logger.debug("Formatting sleep array from sleep windows.")
    sleep_array = np.zeros(len(reference_measure.time), dtype=bool)

    for window in sleep_windows:
        sleep_mask = (reference_measure.time >= window.onset) & (
            reference_measure.time <= window.wakeup
        )
        sleep_array[sleep_mask] = 1

    return sleep_array


def run(
    input: Union[pathlib.Path, str],
    output: Optional[Union[pathlib.Path, str]] = None,
    thresholds: Optional[Tuple[float, float, float]] = None,
    calibrator: Union[
        None,
        Literal["ggir", "gradient"],
    ] = "gradient",
    epoch_length: Union[float, None] = 5,
    activity_metric: Literal["enmo", "mad", "ag_count"] = "enmo",
    nonwear_algorithm: Sequence[Literal["ggir", "cta", "detach"]] = ["ggir"],
    verbosity: int = logging.WARNING,
    output_filetype: Optional[Literal[".csv", ".parquet"]] = None,
) -> Union[models.OrchestratorResults, Dict[str, models.OrchestratorResults]]:
    """Runs main processing steps for wristpy on single files, or directories.

    The run() function will execute the run_file() function on individual files, or
    run_directory() on entire directories. When the input path points to a file, the
    name of the save file will be taken from the given output path (if any). When the
    input path points to a directory the output path must be a valid directory as well.
    Output file names will be derived from original file names in the case of directory
    processing.


    Args:
        input: Path to the input file or directory of files to be read. Currently
            supports .bin and .gt3x
        output: Path to directory data will be saved to. If processing a single file the
            path should end in the save file name in either .csv or .parquet formats.
        thresholds: The cut points for the light, moderate, and vigorous thresholds,
            given in that order. Values must be asscending, unique, and greater than 0.
            Default values are optimized for subjects ages 7-11 [1].
        calibrator: The calibrator to be used on the input data.
        epoch_length: The temporal resolution in seconds, the data will be down sampled
            to. If None is given, and `enmo` is the chosen physical activity metric,
            no down sampling is preformed. Otherwise, for `mad` and `ag_count`, a
            ValueError will be raised.
        activity_metric: The metric to be used for physical activity categorization.
        nonwear_algorithm: The algorithm to be used for nonwear detection.
        verbosity: The logging level for the logger.
        output_filetype: Specifies the data format for the save files. Must be None when
            processing files, must be a valid file type when processing directories.

    Returns:
        All calculated data in a save ready format as a Results object or as a
        dictionary of OrchestratorResults objects.

    Raises:
        ValueError: If the physical activity thresholds are not unique or not in
            ascending order.
        ValueError: If processing a file and the output_filetype is not None
        ValueError: If output is None but output_filetype is not None.

    References:
        [1] Hildebrand, M., et al. (2014). Age group comparability of raw accelerometer
        output from wrist- and hip-worn monitors. Medicine and Science in Sports and
        Exercise, 46(9), 1816-1824.
        [2] Treuth MS, Schmitz K, Catellier DJ, McMurray RG, Murray DM, Almeida MJ,
        Going S, Norman JE, Pate R. Defining accelerometer thresholds for activity
        intensities in adolescent girls. Med Sci Sports Exerc. 2004 Jul;36(7):1259-66.
        PMID: 15235335; PMCID: PMC2423321.
    """
    logger.setLevel(verbosity)

    input = pathlib.Path(input)
    output = pathlib.Path(output) if output is not None else None

    if activity_metric == "enmo":
        thresholds = thresholds or (0.0563, 0.1916, 0.6958)
    elif activity_metric == "mad":
        thresholds = thresholds or (0.029, 0.338, 0.604)
    elif activity_metric == "ag_count":
        thresholds = thresholds or (100, 3000, 5200)

    if not (0 <= thresholds[0] < thresholds[1] < thresholds[2]):
        message = "Threshold values must be >=0, unique, and in ascending order."
        logger.error(message)
        raise ValueError(message)

    if input.is_file():
        if output_filetype is not None:
            raise ValueError(
                "When processing single files, output_filetype should be None - "
                "the file type will be determined from the output path."
            )
        logger.debug("Input is file, forwarding to run_file with output=%s", output)

        return _run_file(
            input=input,
            output=output,
            thresholds=thresholds,
            calibrator=calibrator,
            epoch_length=epoch_length,
            activity_metric=activity_metric,
            verbosity=verbosity,
            nonwear_algorithm=nonwear_algorithm,
        )

    return _run_directory(
        input=input,
        output=output,
        thresholds=thresholds,
        calibrator=calibrator,
        epoch_length=epoch_length,
        verbosity=verbosity,
        output_filetype=output_filetype,
        nonwear_algorithm=nonwear_algorithm,
    )


def _run_directory(
    input: pathlib.Path,
    output: Optional[pathlib.Path] = None,
    thresholds: Tuple[float, float, float] = (0.0563, 0.1916, 0.6958),
    calibrator: Union[
        None,
        Literal["ggir", "gradient"],
    ] = "gradient",
    epoch_length: Union[float, None] = 5,
    nonwear_algorithm: Sequence[Literal["ggir", "cta", "detach"]] = ["ggir"],
    verbosity: int = logging.WARNING,
    output_filetype: Optional[Literal[".csv", ".parquet"]] = None,
) -> Dict[str, models.OrchestratorResults]:
    """Runs main processing steps for wristpy on  directories.

    The run_directory() function will execute the run_file() function on entire
    directories. The input and output (if any) paths must directories. An
    output_filetype must be specified if and only if an output is given. Output file
    names will be derived from input file names.


    Args:
        input: Path to the input directory of files to be read. Currently
            supports .bin and .gt3x
        output: Path to directory data will be saved to.
        thresholds: The cut points for the light, moderate, and vigorous thresholds,
            given in that order. Values must be asscending, unique, and greater than 0.
            Default values are optimized for subjects ages 7-11 [1].
        calibrator: The calibrator to be used on the input data.
        epoch_length: The temporal resolution in seconds, the data will be down sampled
            to. If None is given, and `enmo` is the chosen physical activity metric,
            no down sampling is preformed. Otherwise, for `mad` and `ag_count`, a
            ValueError will be raised.
        nonwear_algorithm: The algorithm to be used for nonwear detection.
        verbosity: The logging level for the logger.
        output_filetype: Specifies the data format for the save files.

    Returns:
        All calculated data in a save ready format as a dictionary of
        OrchestratorResults objects.

    Raises:
        ValueError: The output given is not a directory.
        ValueError: The output_filetype is not a valid type.
        FileNotFoundError: If the input directory contained no files of a valid type.


    References:
        [1] Hildebrand, M., et al. (2014). Age group comparability of raw accelerometer
        output from wrist- and hip-worn monitors. Medicine and Science in Sports and
        Exercise, 46(9), 1816-1824.
    """
    if output is None and output_filetype is not None:
        raise ValueError("If no output is given, output_filetype must be None.")

    if output is not None:
        if output.is_file():
            raise ValueError(
                "Output is a file, but must be a directory when input is a directory."
            )
        if output_filetype not in VALID_FILE_TYPES:
            raise ValueError(
                "Invalid output_filetype: "
                f"{output_filetype}. Valid options are: {VALID_FILE_TYPES}."
            )

    file_names = list(itertools.chain(input.glob("*.gt3x"), input.glob("*.bin")))

    if not file_names:
        raise FileNotFoundError(f"Directory {input} contains no .gt3x or .bin files.")

    results_dict = {}
    for file in file_names:
        output_file_path = (
            output / pathlib.Path(file.stem).with_suffix(output_filetype)  # type: ignore[arg-type] # if output is defined, so is output_filetype
            if output
            else None
        )
        logger.debug(
            "Processing directory: %s, current file: %s, save path: %s",
            input,
            file,
            output_file_path,
        )
        try:
            results_dict[str(file)] = _run_file(
                input=input / file,
                output=output_file_path,
                thresholds=thresholds,
                calibrator=calibrator,
                epoch_length=epoch_length,
                verbosity=verbosity,
                nonwear_algorithm=nonwear_algorithm,
            )
        except Exception as e:
            logger.error("Did not run file: %s, Error: %s", file, e)

    return results_dict


def _run_file(
    input: pathlib.Path,
    output: Optional[pathlib.Path] = None,
    thresholds: Tuple[float, float, float] = (0.0563, 0.1916, 0.6958),
    calibrator: Union[
        None,
        Literal["ggir", "gradient"],
    ] = "gradient",
    epoch_length: Union[float, None] = 5,
    activity_metric: Literal["enmo", "mad", "ag_count"] = "enmo",
    nonwear_algorithm: Sequence[Literal["ggir", "cta", "detach"]] = ["ggir"],
    verbosity: int = logging.WARNING,
) -> models.OrchestratorResults:
    """Runs main processing steps for wristpy and returns data for analysis.

    The run_file() function will provide the user with the specified physical activity
    metric, anglez, physical activity levels, detected sleep periods, and nonwear data.
    All measures will be in the same temporal resolution.
    Users may choose from 'ggir' and 'gradient' calibration methods,
    or enter None to proceed without calibration.

    Args:
        input: Path to the input file to be read. Currently supports .bin and .gt3x
        output: Path to save data to. The path should end in the save file name in
            either .csv or .parquet formats.
        thresholds: The cut points for the light, moderate, and vigorous thresholds,
            given in that order. Values must be ascending, unique, and greater than 0.
            Default values are optimized for subjects ages 7-11 [1].
        calibrator: The calibrator to be used on the input data.
        epoch_length: The temporal resolution in seconds, the data will be down sampled
            to. If None is given, and `enmo` is the chosen physical activity metric,
            no down sampling is preformed. Otherwise, for `mad` and `ag_count`, a
            ValueError will be raised.
        activity_metric: The metric to be used for physical activity categorization.
        nonwear_algorithm: The algorithm to be used for nonwear detection. A sequence of
            algorithms can be provided. If so, a majority vote will be taken.
        verbosity: The logging level for the logger.

    Returns:
        All calculated data in a save ready format as a OrchestratorResults object.

    Raises:
        ValueError: If an invalid Calibrator is chosen
        ValueError: If the detach or CTA algorithms are requested without
            temperature information.

    References:
        [1] Hildebrand, M., et al. (2014). Age group comparability of raw accelerometer
        output from wrist- and hip-worn monitors. Medicine and Science in Sports and
        Exercise, 46(9), 1816-1824.
        [2] Aittasalo, M., Vähä-Ypyä, H., Vasankari, T. et al. Mean amplitude deviation
        calculated from raw acceleration data: a novel method for classifying the
        intensity of adolescents' physical activity irrespective of accelerometer brand.
        BMC Sports Sci Med Rehabil 7, 18 (2015). https://doi.org/10.1186/s13102-015-0010-0.
    """
    logger.setLevel(verbosity)
    if output is not None:
        models.OrchestratorResults.validate_output(output=output)

    if calibrator is not None and calibrator not in ["ggir", "gradient"]:
        msg = (
            f"Invalid calibrator: {calibrator}. Choose: 'ggir', 'gradient'. "
            "Enter None if no calibration is desired."
        )
        logger.error(msg)
        raise ValueError(msg)

    watch_data = readers.read_watch_data(input)

    if calibrator is None:
        logger.debug("Running without calibration")
        calibrated_acceleration = watch_data.acceleration
    else:
        calibrated_acceleration = calibration.CalibrationDispatcher(calibrator).run(
            watch_data.acceleration, return_input_on_error=True
        )

    if watch_data.idle_sleep_mode_flag:
        logger.debug("Imputing idle sleep mode gaps.")
        calibrated_acceleration = (
            idle_sleep_mode_imputation.impute_idle_sleep_mode_gaps(
                calibrated_acceleration
            )
        )

    anglez = metrics.angle_relative_to_horizontal(
        calibrated_acceleration, epoch_length=epoch_length
    )
    activity_measurement = _compute_activity(
        calibrated_acceleration, activity_metric, epoch_length
    )

    sleep_detector = analytics.GgirSleepDetection(anglez)
    sleep_windows = sleep_detector.run_sleep_detection()

    nonwear_array = nonwear_utils.get_nonwear_measurements(
        calibrated_acceleration=calibrated_acceleration,
        temperature=watch_data.temperature,
        non_wear_algorithms=nonwear_algorithm,
    )
    nonwear_epoch = nonwear_utils.nonwear_array_cleanup(
        nonwear_array=nonwear_array,
        reference_measurement=activity_measurement,
        epoch_length=epoch_length,
    )
    physical_activity_levels = analytics.compute_physical_activty_categories(
        activity_measurement, thresholds
    )

    sleep_array = models.Measurement(
        measurements=format_sleep_data(
            sleep_windows=sleep_windows, reference_measure=activity_measurement
        ),
        time=activity_measurement.time,
    )

    results = models.OrchestratorResults(
        physical_activity_metric=activity_measurement,
        anglez=anglez,
        physical_activity_levels=physical_activity_levels,
        sleep_windows_epoch=sleep_array,
        nonwear_epoch=nonwear_epoch,
    )
    if output is not None:
        try:
            results.save_results(output=output)
        except (
            exceptions.InvalidFileTypeError,
            PermissionError,
            FileExistsError,
        ) as exc_info:
            # Allowed to pass to recover in Jupyter Notebook scenarios.
            logger.error(
                (
                    f"Could not save output due to: {exc_info}. Call save_results "
                    " on the output object with a correct filename to save these "
                    "results."
                )
            )

    return results


def _compute_activity(
    acceleration: models.Measurement,
    activity_metric: Literal["ag_count", "mad", "enmo"],
    epoch_length: Union[float, None],
) -> models.Measurement:
    """This is a helper function to organize the computation of the activity metric.

    This function organizes the logic for computing the requested physical activity
    metric at the desired temporal resolution.

    Args:
        acceleration: The acceleration data to compute the activity metric from.
        activity_metric: The metric to be used for physical activity categorization.
        epoch_length: The temporal resolution in seconds, the data will be down sampled
            to.

    Returns:
        A Measurement object with the computed physical activity metric.

    Raises:
        ValueError: If the activity_metric is 'ag_count' or 'mad' and epoch_length is
            None.
    """
    if activity_metric in ("ag_count", "mad") and epoch_length is None:
        raise ValueError("If using 'ag_count' or 'mad', epoch_length must be provided.")

    if activity_metric == "ag_count":
        return metrics.actigraph_activity_counts(
            acceleration,
            epoch_length=epoch_length,  # type: ignore[arg-type] # Guarded by the ValueError statement above.
        )
    elif activity_metric == "mad":
        return metrics.mean_amplitude_deviation(acceleration, epoch_length=epoch_length)  # type: ignore[arg-type] # Guarded by the ValueError statement above.
    return metrics.euclidean_norm_minus_one(acceleration, epoch_length=epoch_length)
