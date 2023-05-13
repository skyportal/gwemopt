import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from gwemopt.io.schedule import read_summary
from gwemopt.run import run

np.random.seed(42)

test_dir = Path(__file__).parent.absolute()
test_data_dir = test_dir.joinpath("data")
expected_results_dir = test_data_dir.joinpath("expected_results")

test_skymap = test_data_dir.joinpath("S190814bv_5_LALInference.v1.fits.gz")


def test_coverage():
    """
    Test coverage

    :return: None
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        telescope_list = ["ZTF", "DECam"]
        schedule_list = [
            os.path.join(expected_results_dir, "schedule_ZTF.dat"),
            os.path.join(expected_results_dir, "schedule_DECam.dat"),
        ]

        # To regenerate the test data, uncomment the following lines
        # temp_dir = Path(__file__).parent.absolute().joinpath("temp")
        # temp_dir.mkdir(exist_ok=True)

        args = [
            f"-t",
            ",".join(telescope_list),
            "-o",
            str(temp_dir),
            "-e",
            str(test_skymap),
            "--doTiles",
            "--doPlots",
            "--doCoverage",
            "--coverageFiles",
            ",".join(schedule_list),
        ]

        run(args)

        new_summary = read_summary(Path(temp_dir).joinpath(f"summary.dat"))
        expected_summary = read_summary(
            expected_results_dir.joinpath(f"summary_coverage.dat")
        )

        pd.testing.assert_frame_equal(
            new_summary.reset_index(drop=True),
            expected_summary.reset_index(drop=True),
        )
