"""Module containing the output classes for writing data to files."""

import datetime
import json
import pathlib
from typing import Any, Dict, Optional, Sequence

import polars as pl
import pydantic

from wristpy.core import config, exceptions, models

VALID_FILE_TYPES = (".csv", ".parquet")

logger = config.get_logger()


class OrchestratorResults(pydantic.BaseModel):
    """Dataclass containing results of orchestrator.run()."""

    physical_activity_metric: Sequence[models.Measurement]
    anglez: models.Measurement
    physical_activity_levels: Sequence[models.Measurement]
    nonwear_status: models.Measurement
    sleep_status: models.Measurement
    sib_periods: models.Measurement
    spt_periods: models.Measurement
    processing_params: Optional[Dict[str, Any]] = None

    def save_results(self, output: pathlib.Path) -> None:
        """Convert to polars and save the dataframe as a csv or parquet file.

        Args:
            output: The path and file name of the data to be saved. as either a csv or
                parquet files.

        """
        logger.debug("Saving results.")
        self.validate_output(output=output)
        output.parent.mkdir(parents=True, exist_ok=True)

        activity_metric_df = pl.DataFrame(
            {
                measurement.name or f"metric_{i}": measurement.measurements
                for i, measurement in enumerate(self.physical_activity_metric)
            }
        )

        physical_activity_levels_df = pl.DataFrame(
            {
                measurement.name
                or f"metric_{i} physical activity levels": measurement.measurements
                for i, measurement in enumerate(self.physical_activity_levels)
            }
        )

        results_dataframe = pl.DataFrame(
            {"time": self.anglez.time}
            | {
                name: value.measurements
                for name, value in self
                if name
                not in [
                    "processing_params",
                    "physical_activity_metric",
                    "physical_activity_levels",
                ]
            }
        )
        full_df = pl.concat(
            [activity_metric_df, physical_activity_levels_df, results_dataframe],
            how="horizontal",
        )
        if output.suffix == ".csv":
            full_df.write_csv(output, separator=",")
        elif output.suffix == ".parquet":
            full_df.write_parquet(output)

        logger.info("Results saved in: %s", output)

        if self.processing_params:
            self.save_config_as_json(output)

    def save_config_as_json(self, output_path: pathlib.Path) -> None:
        """Save processing parameters as a JSON configuration file.

        Args:
            output_path: Path where the data file was saved. The JSON file will use
                the same name but with .json extension.
        """
        if not self.processing_params:
            logger.warning("No processing parameters to save as JSON")
            return

        wristpy_version = config.get_version()

        config_data = {
            "processing_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "wristpy_version": wristpy_version,
            "processing_parameters": self.processing_params,
        }

        config_path = output_path.with_suffix(".json")

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)

        logger.debug("Configuration saved in: %s", config_path)

    @classmethod
    def validate_output(cls, output: pathlib.Path) -> None:
        """Validates that the output path exists and is a valid format.

        Args:
            output: the name of the file to be saved, and the directory it will
                be saved in. Must be a .csv or .parquet file.

        Raises:
            InvalidFileTypeError:If the output file path ends with any extension other
                    than csv or parquet.
        """
        if output.suffix not in VALID_FILE_TYPES:
            raise exceptions.InvalidFileTypeError(
                f"The extension: {output.suffix} is not supported."
                "Please save the file as .csv or .parquet",
            )
