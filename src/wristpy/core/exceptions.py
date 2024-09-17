"""Custom exceptions for wristpy."""

from wristpy.core import config

logger = config.get_logger


class LoggedException(Exception):
    """Base class that automatically logs messages."""

    def __init__(self, message: str) -> None:
        """Initialize a new instance of the LoggedException class.

        Args:
            message: The message to display.
        """
        logger.exception(message)
        super().__init__(message)


class SphereCriteriaError(LoggedException):
    """Data did not meet the sphere criteria."""

    pass


class CalibrationError(LoggedException):
    """Was not able to lower calibration below error threshold."""

    pass


class NoMotionError(LoggedException):
    """No epochs with zero movement could be found in the data."""

    pass


class ZeroScaleError(LoggedException):
    """Scale value went to zero."""

    pass


class InvalidFileTypeError(LoggedException):
    """Wristpy cannot save in the given file type."""

    pass


class DirectoryNotFoundError(LoggedException):
    """Output save path not found."""

    pass
