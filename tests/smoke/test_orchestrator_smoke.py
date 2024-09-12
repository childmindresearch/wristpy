"""Tests the main workflow of wristpy orchestrator."""

import pathlib

import pytest

from wristpy.core import orchestrator


@pytest.mark.parametrize(
    "file_name", [pathlib.Path("test_output.csv"), pathlib.Path("test_output.parquet")]
)
def test_happy_path(
    file_name: pathlib.Path, tmp_path: pathlib.Path
) -> orchestrator.Results:
    """Happy path for orchestrator."""
    test_file = (
        pathlib.Path(__file__).parent.parent / "sample_data" / "example_actigraph.gt3x"
    )
    results = orchestrator.run(input=test_file, output=tmp_path / file_name)

    assert (tmp_path / file_name).exists
    assert isinstance(results, orchestrator.Results)
    assert results.enmo is not None
    assert results.anglez is not None
    assert results.nonwear_epoch1 is not None
    assert results.sleep_windows_epoch1 is not None
    assert results.physical_activity_levels is not None
