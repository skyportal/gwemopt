
import os, sys
import numpy as np
import healpy as hp

from astropy.time import Time

import glue.segments, glue.segmentsUtils

import gwemopt.utils
import gwemopt.rankedTilesGenerator
import gwemopt.moc

def rankedTiles_struct(params,config_struct,telescope):

    nside = params["nside"]

    n_windows = len(params["Tobs"]) // 2
    tot_obs_time = np.sum(np.diff(params["Tobs"])[::2]) * 86400.

    preComputed_256 = os.path.join(params["tilingDir"],'preComputed_%s_pixel_indices_256.dat'%telescope)
    tileObj = gwemopt.rankedTilesGenerator.RankedTileGenerator(params["skymap"],preComputed_256=preComputed_256)
    ranked_tile_index, ranked_tile_probs, ipixs = tileObj.getRankedTiles(resolution=params["nside"])
    ranked_tile_times = tileObj.integrationTime(tot_obs_time, pValTiles=ranked_tile_probs, func=None)
    ranked_tile_times = config_struct["exposuretime"]*np.round(ranked_tile_times/config_struct["exposuretime"])

    tile_struct = {}
    ii = 0
    for index, prob, ipix, exposureTime in zip(ranked_tile_index, ranked_tile_probs, ipixs, ranked_tile_times):
        tile_struct[ii] = {}
        tile_struct[ii]["index"] = index
        tile_struct[ii]["prob"] = prob
        tile_struct[ii]["ipix"] = ipix
        tile_struct[ii]["exposureTime"] = exposureTime
        tile_struct[ii]["ra"] = config_struct["tesselation"][index,1]
        tile_struct[ii]["dec"] = config_struct["tesselation"][index,2]

        if config_struct["FOV_type"] == "square":
            ipix, radecs, patch = gwemopt.utils.getSquarePixels(tile_struct[ii]["ra"], tile_struct[ii]["dec"], config_struct["FOV"], nside)
        elif config_struct["FOV_type"] == "circle":
            ipix, radecs, patch = gwemopt.utils.getCirclePixels(tile_struct[ii]["ra"], tile_struct[ii]["dec"], config_struct["FOV"], nside)

        tile_struct[ii]["ipix"] = ipix
        tile_struct[ii]["corners"] = radecs
        tile_struct[ii]["patch"] = patch        
        ii = ii + 1

    return tile_struct

def rankedTiles(params):

    tile_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]

        tile_struct = rankedTiles_struct(params, config_struct, telescope)
        tile_structs[telescope] = tile_struct 

    return tile_structs

def moc_tiles_struct(params, config_struct, telescope, map_struct, tile_struct):

    n_windows = len(params["Tobs"]) // 2
    tot_obs_time = np.sum(np.diff(params["Tobs"])[::2]) * 86400.

    ranked_tile_probs = gwemopt.moc.compute_moc_map(tile_struct, map_struct["prob"], func='np.sum(x)')
    ranked_tile_times = gwemopt.utils.integrationTime(tot_obs_time, ranked_tile_probs, func=None, T_int=config_struct["exposuretime"])

    keys = tile_struct.keys()
    for key, prob, exposureTime in zip(keys, ranked_tile_probs, ranked_tile_times):
        tile_struct[key]["prob"] = prob
        tile_struct[key]["exposureTime"] = exposureTime

    return tile_struct

def moc(params, map_struct, moc_structs):

    tile_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        moc_struct = moc_structs[telescope]
 
        tile_struct = moc_tiles_struct(params, config_struct, telescope, map_struct, moc_struct)
        tile_structs[telescope] = tile_struct

    return tile_structs
