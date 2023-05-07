import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from gwemopt.read_output import read_schedule
from gwemopt.run import run

np.random.seed(42)

test_dir = Path(__file__).parent.absolute()
test_data_dir = test_dir.joinpath("data")
expected_results_dir = test_data_dir.joinpath("expected_results")


def test_scheduler():
    """
    Test scheduler

    :return: None
    """

    telescope_list = [
        ("ZTF", ["--doReferences"]),
        (
            "KPED",
            ["--tilesType", "galaxy", "--powerlaw_dist_exp", "1.0", "--doCatalog"],
        ),
        # ('TRE', False),
        # ('TNT', False),
        # ("WINTER", False)
    ]

    for telescope, extra in telescope_list:
        with tempfile.TemporaryDirectory() as temp_dir:
            # To regenerate the test data, uncomment the following lines
            # temp_dir = Path(__file__).parent.absolute().joinpath("temp")
            # temp_dir.mkdir(exist_ok=True)

            args = [
                f"-t",
                telescope,
                "-o",
                str(temp_dir),
                "--doSkymap",
                "-n",
                "S190814bv",
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
                "--powerlaw_cl",
                "0.9",
                # "--doMovie"
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
