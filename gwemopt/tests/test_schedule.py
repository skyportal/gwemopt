import glob
import os
import tempfile
from pathlib import Path

import ephem
import numpy as np
import pandas as pd
from astropy import table, time

import gwemopt.catalog
import gwemopt.coverage
import gwemopt.efficiency
import gwemopt.gracedb
import gwemopt.lightcurve
import gwemopt.moc
import gwemopt.plotting
import gwemopt.rankedTilesGenerator
import gwemopt.segments
import gwemopt.tiles
import gwemopt.waw
from gwemopt.paths import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_TILING_DIR,
    REFS_DIR,
    TESSELATION_DIR,
    test_skymap,
)
from gwemopt.read_output import read_schedule
from gwemopt.run import run
from gwemopt.utils import params_checker, read_skymap, readParamsFromFile

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
