import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from gwemopt.io.schedule import read_schedule
from gwemopt.run import run

np.random.seed(42)

test_dir = Path(__file__).parent.absolute()
test_data_dir = test_dir.joinpath("data")
expected_results_dir = test_data_dir.joinpath("expected_results")

test_skymap = test_data_dir.joinpath("S190814bv_5_LALInference.v1.fits.gz")


def test_scheduler():
    """
    Test scheduler

    :return: None
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        telescope_list = [
            (
                "ZTF",
                [
                    "--doReferences",
                    "--doChipGaps",
                    "--doEfficiency",
                    "--doCoverage",
                    "--coverageFiles",
                    os.path.join(temp_dir, "coverage_ZTF.dat"),
                    "--doObservability",
                    "--doMovie",
                ],
            ),
            (
                "KPED",
                [
                    "--tilesType",
                    "galaxy",
                    "--powerlaw_dist_exp",
                    "1.0",
                    "--catalog",
                    "GLADE",
                ],
            ),
            ("DECam", ["--doChipGaps", "--max_nb_tiles", "5", "--doMinimalTiling"]),
            # ('TRE', []),
            # ("WINTER", []),
            # ('TNT', ["--tilesType", "galaxy", "--powerlaw_dist_exp", "1.0", "--doCatalog", "--galaxy_grade", "Sloc"]),
        ]
        for telescope, extra in telescope_list:
            # To regenerate the test data, uncomment the following lines
            # temp_dir = Path(__file__).parent.absolute().joinpath("temp")
            # temp_dir.mkdir(exist_ok=True)

            args = [
                f"-t",
                telescope,
                "-o",
                str(temp_dir),
                "-e",
                str(test_skymap),
                "--doTiles",
                "--doPlots",
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

            new_schedule = read_schedule(
                Path(temp_dir).joinpath(f"schedule_{telescope}.dat")
            )
            expected_schedule = read_schedule(
                expected_results_dir.joinpath(f"schedule_{telescope}.dat")
            )

            pd.testing.assert_frame_equal(
                new_schedule.reset_index(drop=True),
                expected_schedule.reset_index(drop=True),
            )

        # Test the extra efficiency/coverage files

        extra_test_files = [
            "coverage_ZTF.dat",
            "map.dat",
            "summary.dat",
            "efficiency.txt",
            "efficiency_tophat.txt",
        ] + [f"tiles_coverage_int_{t}.txt" for t, _ in telescope_list]

        for extra_test_file in extra_test_files:
            new = pd.read_table(
                Path(temp_dir).joinpath(extra_test_file), delim_whitespace=True
            )
            expected = pd.read_table(
                expected_results_dir.joinpath(extra_test_file), delim_whitespace=True
            )
            pd.testing.assert_frame_equal(
                new.reset_index(drop=True),
                expected.reset_index(drop=True),
            )
