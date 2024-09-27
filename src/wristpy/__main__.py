"""Main function for wristpy."""

from wristpy.core import cli, orchestrator


def run_main() -> orchestrator.Results:
    """Main entry point to wristpy."""
    cli.main()


if __name__ == "__main__":
    cli.main()
