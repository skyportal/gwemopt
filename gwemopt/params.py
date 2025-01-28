import json
from pathlib import Path

import numpy as np
from astropy import time
from astropy import units as u

import gwemopt
from gwemopt.paths import CONFIG_DIR, REFS_DIR, TESSELATION_DIR
from gwemopt.tiles import TILE_TYPES
from gwemopt.utils import tesselation
from typing import Any
from gwemopt.telescope import Telescope

def params_struct(opts) -> tuple[dict[str, Any], list[Telescope], bool]:
    """@Creates gwemopt params structure
    @param opts
        gwemopt command line options
    """

    telescopes = str(opts.telescopes).split(",")

    params = dict(opts.__dict__)

    match params["geometry"]:
        case "2d":
            do_3d = False
        case "3d":
            do_3d = True
        case None:
            do_3d = False
        case _:
            raise ValueError(
                f"The geometry argument from the command-line should be either '2d' or '3d', is {params['geometry']}"
            )

    params["config"] = {}

    telescopes: list[Telescope] = [
        Telescope(
            telescope_name=telescope,
            telescope_description=gwemopt.utils.readParamsFromFile(
                CONFIG_DIR.joinpath(telescope + ".config")
            ),
        )
        for telescope in telescopes
    ]

    params["coverageFiles"] = (
        opts.coverageFiles.split(",") if opts.coverageFiles else None
    )
    params["lightcurveFiles"] = str(opts.lightcurveFiles).split(",")

    params["Tobs"] = np.array(opts.Tobs.split(","), dtype=float)
    params["observedTiles"] = opts.observedTiles.split(",")

    params["treasuremap_status"] = opts.treasuremap_status.split(",")

    params["unbalanced_tiles"] = None
    params["filters"] = opts.filters.split(",")
    params["exposuretimes"] = np.array(opts.exposuretimes.split(","), dtype=float)
    params["catalogDir"] = Path(opts.catalogDir)

    params["ignore_observability"] = (
        opts.ignore_observability if hasattr(opts, "ignore_observability") else False
    )

    params["true_location"] = (
        opts.true_location if hasattr(opts, "true_location") else False
    )

    if hasattr(opts, "true_ra"):
        params["true_ra"] = opts.true_ra

    if hasattr(opts, "true_dec"):
        params["true_dec"] = opts.true_dec

    if hasattr(opts, "true_distance"):
        params["true_distance"] = opts.true_distance

    if params["catalog"] is not None:
        params["galaxy_limit"] = int(opts.galaxy_limit)

    if params["tilesType"] not in TILE_TYPES:
        err = (
            f"Unrecognised tilesType: {params['tilesType']}. "
            f"Accepted values: {TILE_TYPES}"
        )
        raise ValueError(err)

    if opts.start_time is None:
        params["start_time"] = time.Time.now() - time.TimeDelta(1.0 * u.day)
    else:
        params["start_time"] = time.Time(opts.start_time, format="isot", scale="utc")

    if opts.end_time is None:
        params["end_time"] = time.Time.now()
    else:
        params["end_time"] = time.Time(opts.end_time, format="isot", scale="utc")

    params["inclination"] = opts.inclination if hasattr(opts, "inclination") else False

    params["projection"] = (
        opts.projection if hasattr(opts, "projection") else "astro mollweide"
    )

    params["solverType"] = (
        opts.solverType if hasattr(opts, "solverType") else "heuristic"
    )
    params["milpSolver"] = (
        opts.milpSolver if hasattr(opts, "milpSolver") else "PULP_CBC_CMD"
    )
    params["milpOptions"] = (
        json.loads(opts.milpOptions) if hasattr(opts, "milpOptions") else {}
    )

    params["parallelBackend"] = (
        opts.parallelBackend if hasattr(opts, "parallelBackend") else "threading"
    )

    params["movie"] = opts.movie if hasattr(opts, "movie") else False

    params["plots"] = opts.plots.split(",") if hasattr(opts, "plots") else []

    params["confidence_level"] = (
        opts.confidence_level if hasattr(opts, "confidence_level") else 0.9
    )

    return params, telescopes, do_3d
