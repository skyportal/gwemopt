import copy
import os

import healpy as hp
import numpy as np
from joblib import Parallel, delayed
from regions import Regions
from tqdm import tqdm

import gwemopt.tiles
from gwemopt.chipgaps import get_decam_quadrant_ipix, get_ztf_quadrant_ipix
from gwemopt.paths import CONFIG_DIR
from gwemopt.utils.pixels import getCirclePixels, getRegionPixels, getSquarePixels


def create_moc(params, map_struct=None):
    nside = params["nside"]

    if params["doMinimalTiling"]:
        prob = map_struct["prob"]
        cl = params["powerlaw_cl"]
        prob_scaled = copy.deepcopy(prob)
        prob_sorted = np.sort(prob_scaled)[::-1]
        prob_indexes = np.argsort(prob_scaled)[::-1]
        prob_cumsum = np.cumsum(prob_sorted)
        index = np.argmin(np.abs(prob_cumsum - cl)) + 1
        prob_indexes = prob_indexes[: index + 1]

    moc_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tesselation = config_struct["tesselation"]
        moc_struct = {}

        if params["doMinimalTiling"]:
            if config_struct["FOV_type"] == "region" or config_struct["FOV"] < 1.0:
                idxs = hp.pixelfunc.ang2pix(
                    map_struct["nside"],
                    tesselation[:, 1],
                    tesselation[:, 2],
                    lonlat=True,
                )
                isin = np.isin(idxs, prob_indexes)

                idxs = [i for i, x in enumerate(isin) if x]
                print("Keeping %d/%d tiles" % (len(idxs), len(tesselation)))
                tesselation = tesselation[idxs, :]

        if params["doParallel"]:
            moclists = Parallel(
                n_jobs=params["Ncores"],
                backend="multiprocessing",
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
            ipix_keep = map_struct["ipix_keep"]
        else:
            ipix_keep = []
        if params["doMinimalTiling"]:
            moc_struct_new = copy.copy(moc_struct)
            if params["tilesType"] == "galaxy":
                tile_probs = gwemopt.tiles.compute_tiles_map(
                    params, moc_struct_new, prob, func="center", ipix_keep=ipix_keep
                )
            else:
                tile_probs = gwemopt.tiles.compute_tiles_map(
                    params, moc_struct_new, prob, func="np.sum(x)", ipix_keep=ipix_keep
                )

            keys = moc_struct.keys()

            sort_idx = np.argsort(tile_probs)[::-1]
            csm = np.empty(len(tile_probs))
            csm[sort_idx] = np.cumsum(tile_probs[sort_idx])
            ipix_keep = np.where(csm <= cl)[0]

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

    if "rotation" in params:
        rotation = params["rotation"]
    else:
        rotation = None

    if config_struct["FOV_type"] == "square":
        ipix, radecs, patch, area = getSquarePixels(
            ra_pointing, dec_pointing, config_struct["FOV"], nside, rotation=rotation
        )
    elif config_struct["FOV_type"] == "circle":
        ipix, radecs, patch, area = getCirclePixels(
            ra_pointing, dec_pointing, config_struct["FOV"], nside, rotation=rotation
        )
    elif config_struct["FOV_type"] == "region":
        region_file = os.path.join(CONFIG_DIR, config_struct["FOV"])
        regions = Regions.read(region_file, format="ds9")
        ipix, radecs, patch, area = getRegionPixels(
            ra_pointing, dec_pointing, regions, nside, rotation=rotation
        )
    else:
        raise ValueError("FOV_type must be square, circle or region")

    if params["doChipGaps"]:
        if telescope == "ZTF":
            ipixs = get_ztf_quadrant_ipix(nside, ra_pointing, dec_pointing)
            ipix = list({y for x in ipixs for y in x})
        elif telescope == "DECam":
            ipixs = get_decam_quadrant_ipix(nside, ra_pointing, dec_pointing)
            ipix = list({y for x in ipixs for y in x})
        else:
            raise ValueError("Chip gaps only available for DECam and ZTF")

    moc_struct["ra"] = ra_pointing
    moc_struct["dec"] = dec_pointing
    moc_struct["ipix"] = ipix
    moc_struct["corners"] = radecs
    moc_struct["patch"] = patch
    moc_struct["area"] = area

    moc_struct["moc"] = []

    return moc_struct
