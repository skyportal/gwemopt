import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from gwemopt.io.schedule import read_summary
from gwemopt.run import run

_test_dir = Path(__file__).parent.absolute()
_test_data_dir = _test_dir / "data"
expected_results_dir = _test_data_dir / "expected_results"
test_skymap = _test_data_dir / "S190814bv_5_LALInference.v1.fits.gz"


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

        args = [
            "-t",
            ",".join(telescope_list),
            "-o",
            str(temp_dir),
            "-e",
            str(test_skymap),
            "--doTiles",
            "--doCoverage",
            "--coverageFiles",
            ",".join(schedule_list),
        ]

        run(args)

        new_summary = read_summary(Path(temp_dir).joinpath("summary.dat"))
        expected_summary = read_summary(
            expected_results_dir.joinpath("summary_coverage.dat")
        )

        pd.testing.assert_frame_equal(
            new_summary.reset_index(drop=True),
            expected_summary.reset_index(drop=True),
            rtol=1e-2,
        )
