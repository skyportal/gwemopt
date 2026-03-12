import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gwemopt.io.schedule import read_schedule
from gwemopt.run import run

_test_dir = Path(__file__).parent.absolute()
_test_data_dir = _test_dir / "data"
expected_results_dir = _test_data_dir / "expected_results"
test_skymap = _test_data_dir / "S190814bv_5_LALInference.v1.fits.gz"


def _run_telescope(temp_dir, telescope, extra):
    args = [
        "-t",
        telescope,
        "-o",
        str(temp_dir),
        "-e",
        str(test_skymap),
        "--doTiles",
        "--doSchedule",
        "--timeallocationType",
        "powerlaw",
        "--scheduleType",
        "greedy",
        "--filters",
        "g,r",
        "--exposuretimes",
        "30,30",
        "--doSingleExposure",
        "--doAlternatingFilters",
        "--doBalanceExposure",
    ] + extra

    run(args)

    check_files = [
        f"schedule_{telescope}.dat",
        f"tiles_coverage_int_{telescope}.txt",
    ]

    for file_name in check_files:
        print(f"Testing: {file_name}")

        new_schedule = read_schedule(Path(temp_dir).joinpath(file_name))
        expected_schedule = read_schedule(expected_results_dir.joinpath(file_name))

        pd.testing.assert_frame_equal(
            new_schedule.reset_index(drop=True),
            expected_schedule.reset_index(drop=True),
            rtol=1e-2,
        )


def test_scheduler():
    """Test scheduler with ZTF, DECam, and summary output."""

    with tempfile.TemporaryDirectory() as temp_dir:
        Path(temp_dir).joinpath("efficiency.txt").unlink(missing_ok=True)

        _run_telescope(
            temp_dir,
            "ZTF",
            [
                "--doReferences",
                "--doChipGaps",
                "--doEfficiency",
                "--plots",
                "coverage",
                "--doCoverage",
                "--coverageFiles",
                os.path.join(temp_dir, "coverage_ZTF.dat"),
            ],
        )

        _run_telescope(
            temp_dir,
            "DECam",
            ["--max_nb_tiles", "5", "--plots", "coverage"],
        )

        extra_test_files = [
            "summary.dat",
        ]

        for extra_test_file in extra_test_files:
            print(f"Testing: {extra_test_file}")

            new = pd.read_table(Path(temp_dir).joinpath(extra_test_file), sep=r"\s+")
            expected = pd.read_table(
                expected_results_dir.joinpath(extra_test_file), sep=r"\s+"
            )
            pd.testing.assert_frame_equal(
                new.reset_index(drop=True),
                expected.reset_index(drop=True),
                rtol=1e-2,
            )


@pytest.mark.network
def test_scheduler_galaxy():
    """Test scheduler with galaxy tiling (requires GLADE catalog download)."""

    with tempfile.TemporaryDirectory() as temp_dir:
        _run_telescope(
            temp_dir,
            "KPED",
            [
                "--tilesType",
                "galaxy",
                "--plots",
                "coverage",
                "--catalog",
                "GLADE",
            ],
        )
