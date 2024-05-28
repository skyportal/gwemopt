import json
import os
from pathlib import Path

import astroplan
import astropy
import numpy as np
from astropy import table, time
from astropy import units as u

import gwemopt
from gwemopt.paths import CONFIG_DIR, REFS_DIR, TESSELATION_DIR
from gwemopt.tiles import TILE_TYPES


def params_struct(opts):
    """@Creates gwemopt params structure
    @param opts
        gwemopt command line options
    """

    telescopes = str(opts.telescopes).split(",")

    params = dict(opts.__dict__)

    params["config"] = {}
    for telescope in telescopes:
        config_file = CONFIG_DIR.joinpath(telescope + ".config")
        params["config"][telescope] = gwemopt.utils.readParamsFromFile(config_file)
        params["config"][telescope]["telescope"] = telescope
        if opts.doSingleExposure:
            exposuretime = np.array(opts.exposuretimes.split(","), dtype=float)[0]

            params["config"][telescope]["magnitude_orig"] = params["config"][telescope][
                "magnitude"
            ]
            params["config"][telescope]["exposuretime_orig"] = params["config"][
                telescope
            ]["exposuretime"]

            nmag = -2.5 * np.log10(
                np.sqrt(params["config"][telescope]["exposuretime"] / exposuretime)
            )
            params["config"][telescope]["magnitude"] = (
                params["config"][telescope]["magnitude"] + nmag
            )
            params["config"][telescope]["exposuretime"] = exposuretime
        if "tesselationFile" in params["config"][telescope]:
            tessfile = TESSELATION_DIR.joinpath(
                params["config"][telescope]["tesselationFile"]
            )
            if not os.path.isfile(tessfile):
                if params["config"][telescope]["FOV_type"] == "circle":
                    gwemopt.tiles.tesselation_spiral(params["config"][telescope])
                elif params["config"][telescope]["FOV_type"] == "square":
                    gwemopt.tiles.tesselation_packing(params["config"][telescope])
            if opts.tilesType == "galaxy":
                params["config"][telescope]["tesselation"] = np.empty((3,))
            else:
                params["config"][telescope]["tesselation"] = np.loadtxt(
                    tessfile, usecols=(0, 1, 2), comments="%"
                )

        if "referenceFile" in params["config"][telescope]:
            reffile = REFS_DIR.joinpath(params["config"][telescope]["referenceFile"])

            refs = table.unique(
                table.Table.read(reffile, format="ascii", data_start=2, data_end=-1)[
                    "field", "fid"
                ]
            )
            reference_images = {
                group[0]["field"]: group["fid"].astype(int).tolist()
                for group in refs.group_by("field").groups
            }
            reference_images_map = {0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "y"}
            for key in reference_images:
                reference_images[key] = [
                    reference_images_map.get(n, n) for n in reference_images[key]
                ]
            params["config"][telescope]["reference_images"] = reference_images

        location = astropy.coordinates.EarthLocation(
            params["config"][telescope]["longitude"],
            params["config"][telescope]["latitude"],
            params["config"][telescope]["elevation"],
        )
        observer = astroplan.Observer(location=location)
        params["config"][telescope]["observer"] = observer

    params["coverageFiles"] = (
        opts.coverageFiles.split(",") if opts.coverageFiles else None
    )
    params["telescopes"] = telescopes
    params["lightcurveFiles"] = str(opts.lightcurveFiles).split(",")

    params["Tobs"] = np.array(opts.Tobs.split(","), dtype=float)
    params["observedTiles"] = opts.observedTiles.split(",")

    params["treasuremap_status"] = opts.treasuremap_status.split(",")

    if opts.raslice is not None:
        params["raslice"] = np.array(opts.raslice.split(","), dtype=float)
    else:
        params["raslice"] = None

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

    return params
