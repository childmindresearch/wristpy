"""Export data model to CSV files."""

import os

from actigrapy.io.model import ActigraphyData


def export(
    data: ActigraphyData,
    path: str | os.PathLike[str],
) -> None:
    """Export data model to CSV files."""
    data.to_csv(
        path,
        header=True,
        index=False,
    )
