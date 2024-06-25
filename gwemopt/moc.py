import copy
import os

import astropy.units as u
import healpy as hp
import numpy as np
import regions
from astropy.coordinates import SkyCoord
from joblib import Parallel, delayed
from mocpy import MOC
from tqdm import tqdm

import gwemopt.tiles
from gwemopt.chipgaps import get_decam_quadrant_moc, get_ztf_quadrant_moc
from gwemopt.paths import CONFIG_DIR
from gwemopt.utils.parallel import tqdm_joblib
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
                max_depth=np.uint8(10),
                n_threads=n_threads,
            )
        elif config_struct["FOV_type"] == "square":
            mocs = MOC.from_boxes(
                lon=tesselation[:, 1] * u.deg,
                lat=tesselation[:, 2] * u.deg,
                a=config_struct["FOV"] * u.deg,
                b=config_struct["FOV"] * u.deg,
                angle=0 * u.deg,
                max_depth=np.uint8(10),
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


def create_moc(params, map_struct=None):
    nside = params["nside"]

    moc_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tesselation = config_struct["tesselation"]

        if (telescope == "ZTF") and params["doUsePrimary"]:
            idx = np.where(tesselation[:, 0] <= 880)[0]
            tesselation = tesselation[idx, :]
        elif (telescope == "ZTF") and params["doUseSecondary"]:
            idx = np.where(tesselation[:, 0] >= 1000)[0]
            tesselation = tesselation[idx, :]

        moc_struct = construct_moc(params, config_struct, telescope, tesselation)
        if map_struct is not None:
            moc_keep = map_struct["moc_keep"]
        else:
            moc_keep = None

        if params["doMinimalTiling"]:
            moc_struct_new = copy.copy(moc_struct)
            if params["tilesType"] == "galaxy":
                tile_probs = gwemopt.tiles.compute_tiles_map(
                    params,
                    moc_struct_new,
                    map_struct["skymap"],
                    func="center",
                    moc_keep=moc_keep,
                )
            else:
                tile_probs = gwemopt.tiles.compute_tiles_map(
                    params,
                    moc_struct_new,
                    map_struct["skymap"],
                    func="np.sum(x)",
                    moc_keep=moc_keep,
                )

            keys = moc_struct.keys()

            sort_idx = np.argsort(tile_probs)[::-1]
            csm = np.empty(len(tile_probs))
            csm[sort_idx] = np.cumsum(tile_probs[sort_idx])
            tiles_keep = np.where(csm <= params["confidence_level"])[0]

            moc_struct = {}
            cnt = 0
            for ii, key in enumerate(keys):
                if ii in tiles_keep:
                    moc_struct[key] = moc_struct_new[key]
                    cnt = cnt + 1

        moc_structs[telescope] = moc_struct

    return moc_structs
