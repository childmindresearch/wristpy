"""Configuration module for wristpy."""

import logging

import pydantic_settings


class Settings(pydantic_settings.BaseSettings):
    """Settings for wristpy."""

    LIGHT_THRESHOLD: float = 0.03
    MODERATE_THRESHOLD: float = 0.1
    VIGOROUS_THRESHOLD: float = 0.3

    LOGGING_LEVEL: int = logging.INFO

    SHORT_EPOCH_LENGTH: int = 900
    N_SHORT_EPOCH_IN_LONG_EPOCH: int = 4
    STD_CRITERIA: float = 0.013
    RANGE_CRITERIA: float = 0.05


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
