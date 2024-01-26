"""Tests for the fibonacci() function."""
import pytest

from actigrapy import algorithms


def test_fibonacci_success_0() -> None:
    """Test that fibonacci() returns the correct value for valid input."""
    output = algorithms.fibonacci(0)

    assert output == 0


def test_fibonacci_success_18() -> None:
    """Test that fibonacci() returns the correct value for valid input."""
    expected = 4181

    actual = algorithms.fibonacci(19)

    assert actual == expected


def test_fibonacci_negative() -> None:
    """Test that fibonacci() raises an exception for negative input."""
    with pytest.raises(ValueError):
        algorithms.fibonacci(-1)


def test_fibonacci_non_integer() -> None:
    """Test that fibonacci() raises an exception for non-integer input."""
    with pytest.raises(ValueError):
        algorithms.fibonacci(3.14)  # type: ignore[arg-type]
