"""Configuration module for wristpy."""

import pathlib

import pydantic
import pydantic_settings


class Settings(pydantic_settings.BaseSettings):
    """Settings for wristpy."""

    LIGHT_THRESHOLD: float = 0.03
    MODERATE_THRESHOLD: float = 0.1
    VIGOROUS_THRESHOLD: float = 0.3
