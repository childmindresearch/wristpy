"""Main function for wristpy."""

from wristpy.core import cli


def run_main() -> None:
    """Main entry point to wristpy."""
    cli.main()


if __name__ == "__main__":
    cli.main()
