import copy
import os

import astropy.units as u
import numpy as np
import regions
from mocpy import MOC

import gwemopt.tiles
from gwemopt.chipgaps import get_decam_quadrant_moc, get_ztf_quadrant_moc
from gwemopt.paths import CONFIG_DIR
from gwemopt.utils.pixels import get_region_moc


def construct_moc(params, config_struct, telescope, tesselation):
    if params["doParallel"]:
        n_threads = params["Ncores"]
    else:
        n_threads = None

    if params["doChipGaps"]:
        if telescope == "ZTF":
            mocs = get_ztf_quadrant_moc(
                tesselation[:, 1], tesselation[:, 2], n_threads=n_threads
            )
        elif telescope == "DECam":
            mocs = get_decam_quadrant_moc(
                tesselation[:, 1], tesselation[:, 2], n_threads=n_threads
            )
        else:
            raise ValueError("Chip gaps only available for DECam and ZTF")

    else:
        if config_struct["FOV_type"] == "circle":
            mocs = MOC.from_cones(
                lon=tesselation[:, 1] * u.deg,
                lat=tesselation[:, 2] * u.deg,
                radius=config_struct["FOV"] * u.deg,
                max_depth=10,
                n_threads=n_threads,
            )
        elif config_struct["FOV_type"] == "square":
            mocs = MOC.from_boxes(
                lon=tesselation[:, 1] * u.deg,
                lat=tesselation[:, 2] * u.deg,
                a=config_struct["FOV"] * u.deg,
                b=config_struct["FOV"] * u.deg,
                angle=0 * u.deg,
                max_depth=10,
                n_threads=n_threads,
            )
        elif config_struct["FOV_type"] == "region":
            region_file = os.path.join(CONFIG_DIR, config_struct["FOV"])
            region = regions.Regions.read(region_file, format="ds9")
            mocs = get_region_moc(
                tesselation[:, 1], tesselation[:, 2], region, n_threads=n_threads
            )

    return {
        tess[0].astype(int): {"ra": tess[1], "dec": tess[2], "moc": mocs[ii]}
        for ii, tess in enumerate(tesselation)
    }


def create_moc(params, map_struct=None, field_ids=None, from_skyportal=False):
    moc_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tesselation = config_struct["tesselation"]

        if from_skyportal:
            moc_struct = {}
            mocs = []
            for ii, tess in enumerate(tesselation):
                if field_ids is not None:
                    if tess.field_id not in field_ids[telescope]:
                        mocs.append(MOC.new_empty(29))
                        continue
                ranges = np.array(
                    [(tile.healpix.lower, tile.healpix.upper) for tile in tess.tiles]
                )
                moc = MOC.from_depth29_ranges(10, ranges)
                mocs.append(moc)
            for ii, tess in enumerate(tesselation):
                index = tess.field_id

                if (telescope == "ZTF") and params["doUsePrimary"] and (index > 880):
                    continue
                if (telescope == "ZTF") and params["doUseSecondary"] and (index < 1000):
                    continue

                moc = mocs[ii]
                if moc.empty():
                    continue

                moc_struct[index] = {}
                moc_struct[index]["ra"] = tess.ra
                moc_struct[index]["dec"] = tess.dec
                moc_struct[index]["moc"] = moc
        else:
            if (telescope == "ZTF") and params["doUsePrimary"]:
                idx = np.where(tesselation[:, 0] <= 880)[0]
                tesselation = tesselation[idx, :]
            elif (telescope == "ZTF") and params["doUseSecondary"]:
                idx = np.where(tesselation[:, 0] >= 1000)[0]
                tesselation = tesselation[idx, :]

            moc_struct = construct_moc(params, config_struct, telescope, tesselation)

        moc_structs[telescope] = moc_struct

    return moc_structs
