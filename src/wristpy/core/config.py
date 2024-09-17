"""Configuration module for wristpy."""

import logging
import pathlib

import pydantic_settings

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


class Settings(pydantic_settings.BaseSettings):
    """Settings for wristpy."""

    LIGHT_THRESHOLD: float = 0.03
    MODERATE_THRESHOLD: float = 0.1
    VIGOROUS_THRESHOLD: float = 0.3

    LOGGING_LEVEL: int = logging.INFO


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
