"""Configuration module for wristpy."""

import logging
from importlib import metadata

import pydantic_settings
from pydantic import model_validator


class Settings(pydantic_settings.BaseSettings):
    """Settings for wristpy."""

    LIGHT_THRESHOLD: float = 0.03
    MODERATE_THRESHOLD: float = 0.1
    VIGOROUS_THRESHOLD: float = 0.3

    LOGGING_LEVEL: int = logging.INFO

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "Settings":
        """Validate that the thresholds are in ascending order.

        Returns:
            The settings with thresholds in ascending order.

        Raises:
            ValueError if the thresholds are not in ascending order.
        """
        if not (
            0 < self.LIGHT_THRESHOLD < self.MODERATE_THRESHOLD < self.VIGOROUS_THRESHOLD
        ):
            raise ValueError(
                "Light, moderate, and vigorous thresholds must be positive, "
                "unique, and provided in ascending order."
            )
        return self


def get_version() -> str:
    """Return wristpy version."""
    try:
        return metadata.version("wristpy")
    except metadata.PackageNotFoundError:
        return "Version unknown"


def get_logger() -> logging.Logger:
    """Gets the wristpy logger."""
    logger = logging.getLogger("wristpy")
    if logger.handlers:
        return logger

    logger.setLevel(Settings().LOGGING_LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s - %(message)s",  # noqa: E501
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
