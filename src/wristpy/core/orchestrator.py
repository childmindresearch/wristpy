"""Python based runner."""

import itertools
import logging
import pathlib
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

from rich import progress

from wristpy.core import config, exceptions, models
from wristpy.io.readers import readers
from wristpy.io.writers import writers
from wristpy.processing import (
    analytics,
    calibration,
    idle_sleep_mode_imputation,
    metrics,
    processing_utils,
)

logger = config.get_logger()

VALID_FILE_TYPES = (".csv", ".parquet")

DEFAULT_ACTIVITY_THRESHOLDS = {
    "enmo": (0.0563, 0.1916, 0.6958),
    "mad": (0.029, 0.338, 0.604),
    "ag_count": (100, 3000, 5200),
    "mims": (10.558, 15.047, 19.614),
}


def run(
    input: Union[pathlib.Path, str],
    output: Optional[Union[pathlib.Path, str]] = None,
    thresholds: Optional[Sequence[Tuple[float, float, float]]] = None,
    calibrator: Union[
        None,
        Literal["ggir", "gradient"],
    ] = "gradient",
    epoch_length: float = 5,
    activity_metric: Sequence[Literal["enmo", "mad", "ag_count", "mims"]] = ["enmo"],
    nonwear_algorithm: Sequence[Literal["ggir", "cta", "detach"]] = ["ggir"],
    verbosity: int = logging.WARNING,
    output_filetype: Literal[".csv", ".parquet"] = ".csv",
) -> Union[writers.OrchestratorResults, Dict[str, writers.OrchestratorResults]]:
    """Runs main processing steps for wristpy on single files, or directories.

    The run() function will execute the run_file() function on individual files, or
    run_directory() on entire directories. When the input path points to a file, the
    name of the save file will be taken from the given output path (if any). When the
    input path points to a directory the output path must be a valid directory as well.
    Output file names will be derived from original file names in the case of directory
    processing.

    Args:
        input: Path to the input file or directory of files to be read. Currently,
            this supports .bin and .gt3x
        output: Path to directory data will be saved to. If processing a single file the
            path should end in the save file name in either .csv or .parquet formats.
        thresholds: The cut points for the light, moderate, and vigorous thresholds,
            given in that order. One threshold tuple must be provided for each activity
            metric, in the same order the metrics were specified. To use default values
            for all metrics, leave thresholds as None. Values must be ascending, unique,
            and greater than 0. Default values are optimized for subjects ages 7-11 [1]
            [3].
        calibrator: The calibrator to be used on the input data.
        epoch_length: The temporal resolution in seconds, the data will be down sampled
            to. It must be > 0.0.
        activity_metric: The metric(s) to be used for physical activity categorization.
            Multiple metrics can be specified as a sequence.
        nonwear_algorithm: The algorithm to be used for nonwear detection.
        verbosity: The logging level for the logger.
        output_filetype: Specifies the data format for the save files. Only used when
            processing directories.

    Returns:
        All calculated data in a save ready format as a Results object or as a
        dictionary of OrchestratorResults objects.

    Raises:
        ValueError: If the number of physical activity thresholds does not match the
            number of provided activity metrics.
        ValueError: If the physical activity thresholds are not unique or not in
            ascending order.

    References:
        [1] Hildebrand, M., et al. (2014). Age group comparability of raw accelerometer
        output from wrist- and hip-worn monitors. Medicine and Science in Sports and
        Exercise, 46(9), 1816-1824.
        [2] Treuth MS, Schmitz K, Catellier DJ, McMurray RG, Murray DM, Almeida MJ,
        Going S, Norman JE, Pate R. Defining accelerometer thresholds for activity
        intensities in adolescent girls. Med Sci Sports Exerc. 2004 Jul;36(7):1259-66.
        PMID: 15235335; PMCID: PMC2423321.
        [3] Karas M, Muschelli J, Leroux A, Urbanek J, Wanigatunga A, Bai J,
        Crainiceanu C, Schrack J Comparison of Accelerometry-Based Measures of Physical
        Activity: Retrospective Observational Data Analysis Study JMIR Mhealth Uhealth
        2022;10(7):e38077 URL: https://mhealth.jmir.org/2022/7/e38077 DOI: 10.2196/38077
    """
    logger.setLevel(verbosity)

    input = pathlib.Path(input)
    output = pathlib.Path(output) if output is not None else None

    if thresholds is not None:
        if len(activity_metric) != len(thresholds):
            raise ValueError(
                "Number of thresholds did not match the number of activity metrics. "
                "Provide one threshold tuple per metric or use None for defaults."
            )
        metrics_dict = dict(zip(activity_metric, thresholds))
    else:
        metrics_dict = {
            metric: DEFAULT_ACTIVITY_THRESHOLDS[metric] for metric in activity_metric
        }

    for metric, thresh in metrics_dict.items():
        if not (0 <= thresh[0] < thresh[1] < thresh[2]):
            message = (
                f"Invalid thresholds for {metric}."
                f" Threshold values must be >=0, unique, and in ascending order."
            )
            logger.error(message)
            raise ValueError(message)

    if input.is_file():
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
        activity_metric=activity_metric,
        verbosity=verbosity,
        output_filetype=output_filetype,
        nonwear_algorithm=nonwear_algorithm,
    )


def _run_directory(
    input: pathlib.Path,
    output: Optional[pathlib.Path] = None,
    thresholds: Optional[Sequence[Tuple[float, float, float]]] = None,
    calibrator: Union[
        None,
        Literal["ggir", "gradient"],
    ] = "gradient",
    epoch_length: float = 5,
    nonwear_algorithm: Sequence[Literal["ggir", "cta", "detach"]] = ["ggir"],
    verbosity: int = logging.WARNING,
    output_filetype: Literal[".csv", ".parquet"] = ".csv",
    activity_metric: Sequence[Literal["enmo", "mad", "ag_count", "mims"]] = ["enmo"],
) -> Dict[str, writers.OrchestratorResults]:
    """Runs main processing steps for wristpy on directories.

    The _run_directory() function will execute the _run_file() function on entire
    directories. The input and output (if any) paths must be directories. Output file
    names will be derived from input file names.

    Args:
        input: Path to the input directory of files to be read. Currently,
            this supports .bin and .gt3x
        output: Path to directory data will be saved to.
        thresholds: The cut points for the light, moderate, and vigorous thresholds,
            given in that order. One threshold tuple must be provided for each activity
            metric, in the same order the metrics were specified. To use default values
            for all metrics, leave thresholds as None. Values must be ascending, unique,
            and greater than 0. Default values are optimized for subjects ages 7-11
            [1][2].
        calibrator: The calibrator to be used on the input data.
        epoch_length: The temporal resolution in seconds, the data will be down sampled
            to. It must be > 0.0.
        nonwear_algorithm: The algorithm to be used for nonwear detection.
        verbosity: The logging level for the logger.
        output_filetype: Specifies the data format for the save files.
        activity_metric: The metric(s) to be used for physical activity categorization.
            Multiple metrics can be specified as a sequence.

    Returns:
        All calculated data in a save ready format as a dictionary of
        OrchestratorResults objects.

    Raises:
        ValueError: If the output given is not a directory.
        ValueError: If the output_filetype is not a valid type.
        FileNotFoundError: If the input directory contained no files of a valid type.

    References:
        [1] Hildebrand, M., et al. (2014). Age group comparability of raw accelerometer
        output from wrist- and hip-worn monitors. Medicine and Science in Sports and
        Exercise, 46(9), 1816-1824.
        [2] Karas M, Muschelli J, Leroux A, Urbanek J, Wanigatunga A, Bai J,
        Crainiceanu C, Schrack J Comparison of Accelerometry-Based Measures of Physical
        Activity: Retrospective Observational Data Analysis Study JMIR Mhealth Uhealth
        2022;10(7):e38077 URL: https://mhealth.jmir.org/2022/7/e38077 DOI: 10.2196/38077
    """
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
        raise exceptions.EmptyDirectoryError(
            f"Directory {input} contains no .gt3x or .bin files."
        )
    results_dict = {}
    with progress.Progress(
        progress.SpinnerColumn(),
        progress.TextColumn("[progress.description]{task.description}"),
        progress.BarColumn(),
        progress.TaskProgressColumn(),
        console=None,
    ) as progress_bar:
        task = progress_bar.add_task(
            f"[cyan]Processing files in {input.name}...", total=len(file_names)
        )

        for file in file_names:
            output_file_path = (
                output / pathlib.Path(file.stem).with_suffix(output_filetype)
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
                    input=file,
                    output=output_file_path,
                    thresholds=thresholds,
                    calibrator=calibrator,
                    epoch_length=epoch_length,
                    verbosity=verbosity,
                    nonwear_algorithm=nonwear_algorithm,
                    activity_metric=activity_metric,
                )
            except Exception as e:
                logger.error("Did not run file: %s, Error: %s", file, e)
            progress_bar.update(task, advance=1)
    logger.info("Processing for directory %s completed successfully.", input)
    return results_dict


def _run_file(
    input: pathlib.Path,
    output: Optional[pathlib.Path] = None,
    thresholds: Optional[Sequence[Tuple[float, float, float]]] = None,
    calibrator: Union[
        None,
        Literal["ggir", "gradient"],
    ] = "gradient",
    epoch_length: float = 5.0,
    activity_metric: Sequence[Literal["enmo", "mad", "ag_count", "mims"]] = ["enmo"],
    nonwear_algorithm: Sequence[Literal["ggir", "cta", "detach"]] = ["ggir"],
    verbosity: int = logging.WARNING,
) -> writers.OrchestratorResults:
    """Runs main processing steps for wristpy and returns data for analysis.

    The run_file() function will provide the user with the specified physical activity
    metric(s), anglez, physical activity levels, detected sleep periods, and nonwear
    data. All measures will be in the same temporal resolution. Users may choose from
    'ggir' and 'gradient' calibration methods, or enter None to proceed without
    calibration.

    Args:
        input: Path to the input file to be read. Currently, this supports .bin and
            .gt3x
        output: Path to save data to. The path should end in the save file name in
            either .csv or .parquet formats.
        thresholds: The cut points for the light, moderate, and vigorous thresholds,
            given in that order. One threshold tuple must be provided for each activity
            metric, in the same order the metrics were specified. To use default values
            for all metrics, leave thresholds as None. Values must be ascending, unique,
            and greater than 0. Default values are optimized for subjects ages 7-11
            [1]-[3].
        calibrator: The calibrator to be used on the input data.
        epoch_length: The temporal resolution in seconds, the data will be down sampled
            to. It must be > 0.0.
        activity_metric: The metric(s) to be used for physical activity categorization.
            Multiple metrics can be specified as a sequence.
        nonwear_algorithm: The algorithm to be used for nonwear detection. A sequence of
            algorithms can be provided. If so, a majority vote will be taken.
        verbosity: The logging level for the logger.

    Returns:
        All calculated data in a save ready format as a OrchestratorResults object.

    Raises:
        ValueError: If the number of physical activity thresholds does not match the
            number of provided activity metrics.
        ValueError: If the physical activity thresholds are not unique or not in
            ascending order.
        ValueError: If an invalid Calibrator is chosen.
        ValueError: If epoch_length is not greater than 0.

    References:
        [1] Hildebrand, M., et al. (2014). Age group comparability of raw accelerometer
        output from wrist- and hip-worn monitors. Medicine and Science in Sports and
        Exercise, 46(9), 1816-1824.
        [2] Aittasalo, M., Vähä-Ypyä, H., Vasankari, T. et al. Mean amplitude deviation
        calculated from raw acceleration data: a novel method for classifying the
        intensity of adolescents' physical activity irrespective of accelerometer brand.
        BMC Sports Sci Med Rehabil 7, 18 (2015). https://doi.org/10.1186/s13102-015-0010-0.
        [3] Karas M, Muschelli J, Leroux A, Urbanek J, Wanigatunga A, Bai J,
        Crainiceanu C, Schrack J Comparison of Accelerometry-Based Measures of Physical
        Activity: Retrospective Observational Data Analysis Study JMIR Mhealth Uhealth
        2022;10(7):e38077 URL: https://mhealth.jmir.org/2022/7/e38077 DOI: 10.2196/38077
    """
    logger.setLevel(verbosity)
    if output is not None:
        writers.OrchestratorResults.validate_output(output=output)

    if calibrator is not None and calibrator not in ["ggir", "gradient"]:
        msg = (
            f"Invalid calibrator: {calibrator}. Choose: 'ggir', 'gradient'. "
            "Enter None if no calibration is desired."
        )
        logger.error(msg)
        raise ValueError(msg)

    if epoch_length <= 0:
        msg = "Epoch_length must be greater than 0."
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

    metrics_dict: Dict[
        Literal["enmo", "mad", "ag_count", "mims"], Tuple[float, float, float]
    ]
    if thresholds is not None:
        metrics_dict = dict(zip(activity_metric, thresholds))
    else:
        metrics_dict = {
            metric: DEFAULT_ACTIVITY_THRESHOLDS[metric] for metric in activity_metric
        }

    activity_measurements_list = []
    physical_activity_levels_list = []
    for metric_name, thresh in metrics_dict.items():
        metric_measurement = _compute_activity(
            calibrated_acceleration,
            metric_name,
            epoch_length,
            dynamic_range=watch_data.dynamic_range,
        )
        activity_measurements_list.append(metric_measurement)
        physical_activity_levels_list.append(
            analytics.compute_physical_activty_categories(
                metric_measurement,
                thresh,
                name=f"{metric_name} physical activity levels",
            )
        )

    nonwear_array = processing_utils.get_nonwear_measurements(
        calibrated_acceleration=calibrated_acceleration,
        temperature=watch_data.temperature,
        non_wear_algorithms=nonwear_algorithm,
    )

    nonwear_epoch = processing_utils.synchronize_measurements(
        data_measurement=nonwear_array,
        reference_measurement=anglez,
        epoch_length=epoch_length,
    )

    sleep_detector = analytics.GgirSleepDetection(anglez)
    sleep_parameters = sleep_detector.run_sleep_detection()
    sleep_array = analytics.sleep_cleanup(
        sleep_windows=sleep_parameters.sleep_windows, nonwear_measurement=nonwear_epoch
    )
    spt_windows = analytics.sleep_bouts_cleanup(
        sleep_parameter=sleep_parameters.spt_windows,
        nonwear_measurement=nonwear_epoch,
        time_reference_measurement=anglez,
        epoch_length=epoch_length,
    )
    sib_periods = analytics.sleep_bouts_cleanup(
        sleep_parameter=sleep_parameters.sib_periods,
        nonwear_measurement=nonwear_epoch,
        time_reference_measurement=anglez,
        epoch_length=epoch_length,
    )

    parameters_dictionary = {
        "thresholds": list(thresholds) if thresholds is not None else None,
        "calibrator": calibrator,
        "epoch_length": epoch_length,
        "activity_metric": activity_metric,
        "nonwear_algorithm": list(nonwear_algorithm),
        "input_file": str(input),
        "time_zone": watch_data.time_zone,
    }

    results = writers.OrchestratorResults(
        physical_activity_metric=activity_measurements_list,
        anglez=anglez,
        physical_activity_levels=physical_activity_levels_list,
        sleep_status=sleep_array,
        sib_periods=sib_periods,
        spt_periods=spt_windows,
        nonwear_status=nonwear_epoch,
        processing_params=parameters_dictionary,
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
                    "Could not save output due to: %s. Call save_results "
                    "on the output object with a correct filename to save these "
                    "results.",
                    exc_info,
                )
            )
    logger.info("Processing for %s completed successfully.", input.stem)
    return results


def _compute_activity(
    acceleration: models.Measurement,
    activity_metric: Literal["ag_count", "mad", "enmo", "mims"],
    epoch_length: float,
    dynamic_range: Optional[tuple[float, float]],
) -> models.Measurement:
    """This is a helper function to organize the computation of the activity metric.

    This function organizes the logic for computing the requested physical activity
    metric at the desired temporal resolution.

    Args:
        acceleration: The acceleration data to compute the activity metric from.
        activity_metric: The metric to be used for physical activity categorization.
        epoch_length: The temporal resolution in seconds, the data will be down sampled
            to.
        dynamic_range: Tuple of the minimum and maximum accelerometer values. This
            argument is only relevant to the mims metric. Values are taken from watch
            metadata, if no metadata could be extracted, the default
            values of (-8,8) are used.

    Returns:
        A Measurement object with the computed physical activity metric.
    """
    if activity_metric == "ag_count":
        return metrics.actigraph_activity_counts(
            acceleration, epoch_length=epoch_length, name=activity_metric
        )
    elif activity_metric == "mad":
        return metrics.mean_amplitude_deviation(
            acceleration, epoch_length=epoch_length, name=activity_metric
        )
    elif activity_metric == "mims":
        if dynamic_range is None:
            return metrics.monitor_independent_movement_summary_units(
                acceleration, epoch=epoch_length, name=activity_metric
            )
        return metrics.monitor_independent_movement_summary_units(
            acceleration,
            epoch=epoch_length,
            dynamic_range=dynamic_range,
            name=activity_metric,
        )

    return metrics.euclidean_norm_minus_one(
        acceleration, epoch_length=epoch_length, name=activity_metric
    )
