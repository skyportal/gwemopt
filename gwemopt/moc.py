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
from gwemopt.telescope import Telescope


def construct_moc(params, telescope: Telescope, tesselation):
    if params["doParallel"]:
        n_threads = params["Ncores"]
    else:
        n_threads = None

    if params["doChipGaps"]:
        if telescope.telescope_name == "ZTF":
            mocs = get_ztf_quadrant_moc(
                tesselation[:, 1], tesselation[:, 2], n_threads=n_threads
            )
        elif telescope.telescope_name == "DECam":
            mocs = get_decam_quadrant_moc(
                tesselation[:, 1], tesselation[:, 2], n_threads=n_threads
            )
        else:
            raise ValueError("Chip gaps only available for DECam and ZTF")

    else:
        if telescope.fov_type == "circle":
            mocs = MOC.from_cones(
                lon=tesselation[:, 1] * u.deg,
                lat=tesselation[:, 2] * u.deg,
                radius=telescope.fov * u.deg,
                max_depth=10,
                n_threads=n_threads,
            )
        elif telescope.fov_type == "square":
            mocs = MOC.from_boxes(
                lon=tesselation[:, 1] * u.deg,
                lat=tesselation[:, 2] * u.deg,
                a=telescope.fov * u.deg,
                b=telescope.fov * u.deg,
                angle=0 * u.deg,
                max_depth=10,
                n_threads=n_threads,
            )
        elif telescope.fov_type == "region":
            region_file = os.path.join(CONFIG_DIR, telescope.fov)
            region = regions.Regions.read(region_file, format="ds9")
            mocs = get_region_moc(
                tesselation[:, 1], tesselation[:, 2], region, n_threads=n_threads
            )

    return {
        tess[0].astype(int): {"ra": tess[1], "dec": tess[2], "moc": mocs[ii]}
        for ii, tess in enumerate(tesselation)
    }


def create_moc(params, telescopes: list[Telescope], map_struct=None, field_ids=None, from_skyportal=False):
    moc_structs = {}
    for telescope in telescopes:
        if from_skyportal:
            moc_struct = {}
            mocs = []
            for ii, tess in enumerate(telescope.tesselation):
                if field_ids is not None:
                    if tess.field_id not in field_ids[telescope.telescope_name]:
                        mocs.append(MOC.new_empty(29))
                        continue
                ranges = np.array(
                    [(tile.healpix.lower, tile.healpix.upper) for tile in tess.tiles]
                )
                moc = MOC.from_depth29_ranges(10, ranges)
                mocs.append(moc)
            for ii, tess in enumerate(telescope.tesselation):
                index = tess.field_id

                if (telescope.telescope_name == "ZTF") and params["doUsePrimary"] and (index > 880):
                    continue
                if (telescope.telescope_name == "ZTF") and params["doUseSecondary"] and (index < 1000):
                    continue

                moc = mocs[ii]
                if moc.empty():
                    continue

                moc_struct[index] = {}
                moc_struct[index]["ra"] = tess.ra
                moc_struct[index]["dec"] = tess.dec
                moc_struct[index]["moc"] = moc
        else:
            tesselation = telescope.tesselation
            if (telescope.telescope_name == "ZTF") and params["doUsePrimary"]:
                idx = np.where(tesselation[:, 0] <= 880)[0]
                tesselation = tesselation[idx, :]
            elif (telescope.telescope_name == "ZTF") and params["doUseSecondary"]:
                idx = np.where(tesselation[:, 0] >= 1000)[0]
                tesselation = tesselation[idx, :]

            moc_struct = construct_moc(params, telescope, tesselation)

        moc_structs[telescope.telescope_name] = moc_struct

    return moc_structs
