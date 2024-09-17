"""Configuration module for wristpy."""

import logging
from typing import List

import pydantic_settings
from pydantic import field_validator


class Settings(pydantic_settings.BaseSettings):
    """Settings for wristpy."""

    LIGHT_THRESHOLD: float = 0.03
    MODERATE_THRESHOLD: float = 0.1
    VIGOROUS_THRESHOLD: float = 0.3

    LOGGING_LEVEL: int = logging.INFO

    @field_validator("LIGHT_THRESHOLD", "MODERATE_THRESHOLD", "VIGOROUS_THRESHOLD")
    def validate_threshold_order(cls, v: List[float]) -> List[float]:
        """Validate that the thresholds are in ascending order.

        Args:
            cls: The class.
            v: the three thresholds we will evalute.

        Returns:
            The thresholds where light < moderate < vigorous.
        """


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
