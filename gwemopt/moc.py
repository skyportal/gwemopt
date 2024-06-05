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
from gwemopt.utils.pixels import getRegionPixels


def create_moc(params, map_struct=None):
    nside = params["nside"]

    moc_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tesselation = config_struct["tesselation"]
        moc_struct = {}

        if params["doChipGaps"]:
            if params["doParallel"]:
                n_threads = params["Ncores"]
            else:
                n_threads = None
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

            for ii, tess in enumerate(tesselation):
                index, ra, dec = tess[0], tess[1], tess[2]
                if (telescope == "ZTF") and params["doUsePrimary"] and (index > 880):
                    continue
                if (telescope == "ZTF") and params["doUseSecondary"] and (index < 1000):
                    continue
                index = index.astype(int)
                moc_struct[index] = {"ra": ra, "dec": dec, "moc": mocs[ii]}
        else:
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
                    if (
                        (telescope == "ZTF")
                        and params["doUsePrimary"]
                        and (index > 880)
                    ):
                        continue
                    if (
                        (telescope == "ZTF")
                        and params["doUseSecondary"]
                        and (index < 1000)
                    ):
                        continue
                    moc_struct[index] = moclists[ii]
            else:
                for ii, tess in tqdm(enumerate(tesselation), total=len(tesselation)):
                    index, ra, dec = tess[0], tess[1], tess[2]
                    if (
                        (telescope == "ZTF")
                        and params["doUsePrimary"]
                        and (index > 880)
                    ):
                        continue
                    if (
                        (telescope == "ZTF")
                        and params["doUseSecondary"]
                        and (index < 1000)
                    ):
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
            tiles_keep = np.where(csm <= params["confidence_level"])[0]

            moc_struct = {}
            cnt = 0
            for ii, key in enumerate(keys):
                if ii in tiles_keep:
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
        center = SkyCoord(ra_pointing, dec_pointing, unit="deg", frame="icrs")
        region = regions.RectangleSkyRegion(
            center, config_struct["FOV"] * u.deg, config_struct["FOV"] * u.deg
        )
        moc = MOC.from_astropy_regions(region, max_depth=10)
    elif config_struct["FOV_type"] == "circle":
        center = SkyCoord(ra_pointing, dec_pointing, unit="deg", frame="icrs")
        region = regions.CircleSkyRegion(center, radius=config_struct["FOV"] * u.deg)
        moc = MOC.from_astropy_regions(region, max_depth=10)
    elif config_struct["FOV_type"] == "region":
        region_file = os.path.join(CONFIG_DIR, config_struct["FOV"])
        region = regions.Regions.read(region_file, format="ds9")
        moc = getRegionPixels(
            ra_pointing,
            dec_pointing,
            region,
            nside,
        )
    else:
        raise ValueError("FOV_type must be square, circle or region")

    moc_struct["ra"] = ra_pointing
    moc_struct["dec"] = dec_pointing
    moc_struct["moc"] = moc

    return moc_struct
