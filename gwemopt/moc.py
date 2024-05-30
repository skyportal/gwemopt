import copy
import os

import healpy as hp
import numpy as np
from joblib import Parallel, delayed
from regions import Regions
from tqdm import tqdm

import gwemopt.tiles
from gwemopt.chipgaps import get_decam_quadrant_moc, get_ztf_quadrant_moc
from gwemopt.paths import CONFIG_DIR
from gwemopt.utils.parallel import tqdm_joblib
from gwemopt.utils.pixels import getCirclePixels, getRegionPixels, getSquarePixels


def create_moc(params, map_struct=None):
    nside = params["nside"]

    moc_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tesselation = config_struct["tesselation"]
        moc_struct = {}

        if params["doParallel"]:
            with tqdm_joblib(
                tqdm(desc="MOC creation", total=len(tesselation))
            ) as progress_bar:
                moclists = Parallel(
                    n_jobs=params["Ncores"],
                    backend=params["parallelBackend"],
                    batch_size=int(len(tesselation) / params["Ncores"]) + 1,
                )(
                    delayed(Fov2Moc)(
                        params, config_struct, telescope, tess[1], tess[2], nside
                    )
                    for tess in tesselation
                )
            for ii, tess in tqdm(enumerate(tesselation), total=len(tesselation)):
                index, ra, dec = tess[0], tess[1], tess[2]
                if (telescope == "ZTF") and params["doUsePrimary"] and (index > 880):
                    continue
                if (telescope == "ZTF") and params["doUseSecondary"] and (index < 1000):
                    continue
                moc_struct[index] = moclists[ii]
        else:
            for ii, tess in tqdm(enumerate(tesselation), total=len(tesselation)):
                index, ra, dec = tess[0], tess[1], tess[2]
                if (telescope == "ZTF") and params["doUsePrimary"] and (index > 880):
                    continue
                if (telescope == "ZTF") and params["doUseSecondary"] and (index < 1000):
                    continue
                index = index.astype(int)
                moc_struct[index] = Fov2Moc(
                    params, config_struct, telescope, ra, dec, nside
                )

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
            ipix_keep = np.where(csm <= params["powerlaw_cl"])[0]

            moc_struct = {}
            cnt = 0
            for ii, key in enumerate(keys):
                if ii in ipix_keep:
                    moc_struct[key] = moc_struct_new[key]
                    cnt = cnt + 1

        moc_structs[telescope] = moc_struct

    return moc_structs


def Fov2Moc(params, config_struct, telescope, ra_pointing, dec_pointing, nside):
    """Return a MOC in fits file of a fov footprint.
    The MOC fov is displayed in real time in an Aladin plan.

    Input:
        ra--> right ascention of fov center [deg]
        dec --> declination of fov center [deg]
        fov_width --> fov width [deg]
        fov_height --> fov height [deg]
        nside --> healpix resolution; by default
    """

    moc_struct = {}

    if config_struct["FOV_type"] == "square":
        moc = getSquarePixels(
            ra_pointing,
            dec_pointing,
            config_struct["FOV"],
        )
    elif config_struct["FOV_type"] == "circle":
        moc = getCirclePixels(
            ra_pointing,
            dec_pointing,
            config_struct["FOV"],
        )
    elif config_struct["FOV_type"] == "region":
        region_file = os.path.join(CONFIG_DIR, config_struct["FOV"])
        regions = Regions.read(region_file, format="ds9")
        moc = getRegionPixels(
            ra_pointing,
            dec_pointing,
            regions,
            nside,
        )
    else:
        raise ValueError("FOV_type must be square, circle or region")

    if params["doChipGaps"]:
        if telescope == "ZTF":
            moc = get_ztf_quadrant_moc(ra_pointing, dec_pointing)
        elif telescope == "DECam":
            moc = get_decam_quadrant_moc(ra_pointing, dec_pointing)
        else:
            raise ValueError("Chip gaps only available for DECam and ZTF")

    moc_struct["ra"] = ra_pointing
    moc_struct["dec"] = dec_pointing
    moc_struct["moc"] = moc

    return moc_struct
