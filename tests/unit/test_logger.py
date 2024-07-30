"""Test logging in config.py."""

import pytest

from wristpy.core import config


def test_get_logger(caplog: pytest.LogCaptureFixture) -> None:
    """Test the wristpy logger with level set to 20 (info)."""
    logger = config.get_logger()

    logger.debug("Debug message here.")
    logger.info("Info message here.")
    logger.warning("Warning message here.")

    assert logger.getEffectiveLevel() == 20
    assert "Debug message here" not in caplog.text
    assert "Info message here." in caplog.text
    assert "Warning message here." in caplog.text
