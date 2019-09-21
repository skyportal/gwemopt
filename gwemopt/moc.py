import time
import copy

from mocpy import MOC
from astropy.table import Table
from joblib import Parallel, delayed
import healpy as hp
import numpy as np

import shapely.geometry

import gwemopt.utils
import gwemopt.tiles
import gwemopt.ztf_tiling

def create_moc(params, map_struct=None):

    nside = params["nside"]
    npix = hp.nside2npix(nside)

    if params["doMinimalTiling"]:
        prob = map_struct["prob"]

        n, cl, dist_exp = params["powerlaw_n"], params["powerlaw_cl"], params["powerlaw_dist_exp"]
        prob_scaled = copy.deepcopy(prob)
        prob_sorted = np.sort(prob_scaled)[::-1]
        prob_indexes = np.argsort(prob_scaled)[::-1]
        prob_cumsum = np.cumsum(prob_sorted)
        index = np.argmin(np.abs(prob_cumsum - cl)) + 1
        prob_indexes = prob_indexes[:index+1]

    if "doUsePrimary" in params:
        doUsePrimary = params["doUsePrimary"]
    else:
        doUsePrimary = False

    moc_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tesselation = config_struct["tesselation"]
        moc_struct = {}

        if params["doMinimalTiling"] and (config_struct["FOV"] < 1.0):
            idxs = hp.pixelfunc.ang2pix(map_struct["nside"], tesselation[:,1], tesselation[:,2], lonlat=True)
            isin = np.isin(idxs, prob_indexes)
  
            idxs = [i for i, x in enumerate(isin) if x] 
            print("Keeping %d/%d tiles" % (len(idxs), len(tesselation)))
            tesselation = tesselation[idxs,:]

        if params["doParallel"]:
            moclists = Parallel(n_jobs=params["Ncores"])(delayed(Fov2Moc)(params, config_struct, telescope, tess[1], tess[2], nside) for tess in tesselation)
            for ii, tess in enumerate(tesselation):
                index, ra, dec = tess[0], tess[1], tess[2]
                if (telescope == "ZTF") and doUsePrimary and (index > 880):
                    continue
                moc_struct[index] = moclists[ii]    
        else:
            for ii, tess in enumerate(tesselation):
                index, ra, dec = tess[0], tess[1], tess[2]
                if (telescope == "ZTF") and doUsePrimary and (index > 880):
                    continue
                index = index.astype(int)
                moc_struct[index] = Fov2Moc(params, config_struct, telescope, ra, dec, nside)

        if map_struct is not None:
             ipix_keep = map_struct["ipix_keep"]
        else:
            ipix_keep = []
        if params["doMinimalTiling"]:
            moc_struct_new = copy.deepcopy(moc_struct)
            if params["tilesType"] == "galaxy":
                tile_probs = gwemopt.tiles.compute_tiles_map(params, moc_struct_new, prob, func='center', ipix_keep=ipix_keep)
            else:
                tile_probs = gwemopt.tiles.compute_tiles_map(params, moc_struct_new, prob, func='np.sum(x)', ipix_keep=ipix_keep)

            keys = moc_struct.keys()

            sort_idx = np.argsort(tile_probs)[::-1]
            csm = np.empty(len(tile_probs))
            csm[sort_idx] = np.cumsum(tile_probs[sort_idx])
            ipix_keep = np.where(csm <= cl)[0]

            probs = []
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
        rotation=params["rotation"]
    else:
        rotation=None
 
    if config_struct["FOV_type"] == "square": 
        ipix, radecs, patch, area = gwemopt.utils.getSquarePixels(ra_pointing, dec_pointing, config_struct["FOV"], nside, rotation=rotation)
    elif config_struct["FOV_type"] == "circle":
        ipix, radecs, patch, area = gwemopt.utils.getCirclePixels(ra_pointing, dec_pointing, config_struct["FOV"], nside, rotation=rotation)

    if params["doChipGaps"]:
        if telescope == "ZTF":
            ipixs = gwemopt.ztf_tiling.get_quadrant_ipix(nside, ra_pointing, dec_pointing)
            ipix = list({y for x in ipixs for y in x})
        #else:
        #    print("Requested chip gaps with non-ZTF detector, will use moc.")

    moc_struct["ra"] = ra_pointing
    moc_struct["dec"] = dec_pointing
    moc_struct["ipix"] = ipix
    moc_struct["corners"] = radecs
    moc_struct["patch"] = patch
    moc_struct["area"] = area

    if False:
    #if len(ipix) > 0:
        # from index to polar coordinates
        theta, phi = hp.pix2ang(nside, ipix)

        # converting these to right ascension and declination in degrees
        ra = np.rad2deg(phi)
        dec = np.rad2deg(0.5 * np.pi - theta)

        box_ipix = Table([ra, dec], names = ('RA[deg]', 'DEC[deg]'),
                 meta = {'ipix': 'ipix table'})

        moc_order = int(np.log(nside)/ np.log(2))
        moc = MOC.from_table( box_ipix, 'RA[deg]', 'DEC[deg]', moc_order )

        moc_struct["moc"] = moc
    else:
        moc_struct["moc"] = []

    return moc_struct

