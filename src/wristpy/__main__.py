"""Main function for wristpy."""

from wristpy.core import cli, orchestrator


def run_main() -> orchestrator.Results:
    """Main entry point to wristpy.

    Returns:
        A Results object containing enmo, anglez, physical activity levels, nonwear
        detection, and sleep detection.
    """
    cli.main()


if __name__ == "__main__":
    cli.main()
