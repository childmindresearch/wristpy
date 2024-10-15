"""Configuration module for wristpy."""

import logging
from importlib import metadata


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

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s - %(message)s",  # noqa: E501
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
