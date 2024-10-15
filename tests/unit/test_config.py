"""Test logging in config.py."""

import logging

import pytest

from wristpy.core import config


def test_get_logger(caplog: pytest.LogCaptureFixture) -> None:
    """Test the wristpy logger with level set to default 20 (info)."""
    if logging.getLogger("wristpy").handlers:
        logging.getLogger("wristpy").handlers.clear()
    logger = config.get_logger()

    logger.debug("Debug message here.")
    logger.info("Info message here.")
    logger.warning("Warning message here.")

    assert logger.getEffectiveLevel() == 20
    assert "Debug message here" not in caplog.text
    assert "Info message here." in caplog.text
    assert "Warning message here." in caplog.text


def test_get_logger_second_call() -> None:
    """Test get logger when a handler already exists."""
    logger = config.get_logger()
    second_logger = config.get_logger()

    assert len(logger.handlers) == len(second_logger.handlers) == 1
    assert logger.handlers[0] is second_logger.handlers[0]
    assert logger is second_logger
