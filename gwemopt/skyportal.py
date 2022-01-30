import time
import copy

import mocpy
from mocpy import MOC
from astropy.table import Table
import healpy as hp
import numpy as np

import ligo.segments as segments
from ligo.skymap.bayestar import rasterize

import gwemopt.tiles

def create_moc_from_skyportal(params, map_struct=None):

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

    if "doUseSecondary" in params:
        doUseSecondary = params["doUseSecondary"]
    else:
        doUseSecondary = False

    npix = hp.nside2npix(nside)
    pixarea = hp.nside2pixarea(nside)

    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra = np.rad2deg(phi)*24.0/360.0
    dec = np.rad2deg(0.5*np.pi - theta)

    moc_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tesselation = config_struct["tesselation"]
        moc_struct = {}

        if params["doParallel"]:
            ipixs = Parallel(n_jobs=params["Ncores"])(delayed(skyportal2Moc)(tess, nside) for tess in tesselation)
        else:
            ipixs = []
            for ii, tess in enumerate(tesselation):
                ipixs.append(skyportal2FOV(tess, nside))

        for ii, tess in enumerate(tesselation):
            index = tess.field_id

            if (telescope == "ZTF") and doUsePrimary and (index > 880):
                continue
            if (telescope == "ZTF") and doUseSecondary and (index < 1000):
                continue

            ipix = ipixs[ii]
            if len(ipix) == 0: continue

            moc_struct[index] = {}
            moc_struct[index]["ra"] = np.median(ra[ipix])
            moc_struct[index]["dec"] = np.median(dec[ipix])
            moc_struct[index]["ipix"] = ipix
            moc_struct[index]["corners"] = [[np.min(ra[ipix]), np.min(dec[ipix])],
                                            [np.min(ra[ipix]), np.max(dec[ipix])],
                                            [np.max(ra[ipix]), np.max(dec[ipix])],
                                            [np.max(ra[ipix]), np.min(dec[ipix])]]
            moc_struct[index]["patch"] = []
            moc_struct[index]["area"] = len(ipix)*pixarea

        if map_struct is not None:
            ipix_keep = map_struct["ipix_keep"]
        else:
            ipix_keep = []

        if params["doMinimalTiling"]:
            moc_struct_new = copy.copy(moc_struct)
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


def skyportal2FOV(tess, nside):

    moc = moc_from_tiles([tile.healpix for tile in tess.tiles], 2**29)
    pix_id = mocpy.mocpy.flatten_pixels(moc._interval_set._intervals, int(np.log2(nside)))
    if len(pix_id) > 0:
        ipix = hp.nest2ring(int(nside), pix_id.tolist())
    else:
        ipix = []

    return ipix

def moc_from_tiles(rangeSet, nside):
    depth = int(np.log2(nside))
    segmentlist = segments.segmentlist()
    for x in rangeSet:
        segment = segments.segment(x.lower, x.upper-1)
        segmentlist = segmentlist + segments.segmentlist([segment])
    segmentlist.coalesce()

    MOCstr = f'{depth}/' + ' '.join(map(lambda x: f'{x[0]}-{x[1]}',
                                        segmentlist))
    return MOC.from_string(MOCstr)
