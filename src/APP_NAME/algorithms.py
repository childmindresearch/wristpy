"""Example functions for APP_NAME."""


def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number.

    Args:
        n: The index of the Fibonacci number to return.

    Returns:
        The n-th Fibonacci number.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    if int(n) != n:
        raise ValueError("n must be an integer")

    if n == 0:
        return 0
    if n <= 2:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)
