
import os, sys
import numpy as np
import healpy as hp

from scipy.stats import norm

from astropy.time import Time

import glue.segments, glue.segmentsUtils

import gwemopt.utils
import gwemopt.rankedTilesGenerator
import gwemopt.moc

try:
    import gwemopt.multinest
except:
    print "No Multinest present."

def greedy(params, map_struct):

    tile_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]

        tile_struct = gwemopt.multinest.greedy_tiles_struct(params, config_struct, telescope, map_struct)
        tile_structs[telescope] = tile_struct

    return tile_structs

def rankedTiles_struct(params,config_struct,telescope,map_struct):

    nside = params["nside"]

    n_windows = len(params["Tobs"]) // 2
    tot_obs_time = np.sum(np.diff(params["Tobs"])[::2]) * 86400.

    preComputed_256 = os.path.join(params["tilingDir"],'preComputed_%s_pixel_indices_256.dat'%telescope)
    tileObj = gwemopt.rankedTilesGenerator.RankedTileGenerator(params["skymap"],preComputed_256=preComputed_256)

    if "observability" in map_struct:
        tileObj.skymap = map_struct["observability"][telescope]["prob"]
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
            ipix, radecs, patch, area = gwemopt.utils.getSquarePixels(tile_struct[ii]["ra"], tile_struct[ii]["dec"], config_struct["FOV"], nside)
        elif config_struct["FOV_type"] == "circle":
            ipix, radecs, patch, area = gwemopt.utils.getCirclePixels(tile_struct[ii]["ra"], tile_struct[ii]["dec"], config_struct["FOV"], nside)

        tile_struct[ii]["ipix"] = ipix
        tile_struct[ii]["corners"] = radecs
        tile_struct[ii]["patch"] = patch  
        tile_struct[ii]["area"] = area      
        ii = ii + 1

    return tile_struct

def rankedTiles(params, map_struct):

    tile_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]

        tile_struct = rankedTiles_struct(params, config_struct, telescope, map_struct)
        tile_structs[telescope] = tile_struct 

    return tile_structs

def waterfall_tiles_struct(params, config_struct, telescope, map_struct, tile_struct):

    n_windows = len(params["Tobs"]) // 2
    tot_obs_time = np.sum(np.diff(params["Tobs"])[::2]) * 86400.

    if "observability" in map_struct:
        prob = map_struct["observability"][telescope]["prob"]
    else:
        prob = map_struct["prob"]
 
    ranked_tile_probs = compute_tiles_map(tile_struct, prob, func='np.sum(x)')
    ranked_tile_times = gwemopt.utils.integrationTime(tot_obs_time, ranked_tile_probs, func=None, T_int=config_struct["exposuretime"])

    keys = tile_struct.keys()
    for key, prob, exposureTime in zip(keys, ranked_tile_probs, ranked_tile_times):
        tile_struct[key]["prob"] = prob
        tile_struct[key]["exposureTime"] = exposureTime

    return tile_struct

def moc(params, map_struct, tile_structs):

    tile_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tile_struct = tile_structs[telescope]
 
        tile_struct = waterfall_tiles_struct(params, config_struct, telescope, map_struct, tile_struct)
        tile_structs[telescope] = tile_struct

    return tile_structs

def pem_tiles_struct(params, config_struct, telescope, map_struct, tile_struct):

    n_windows = len(params["Tobs"]) // 2
    tot_obs_time = np.sum(np.diff(params["Tobs"])[::2]) * 86400.

    if "observability" in map_struct:
        prob = map_struct["observability"][telescope]["prob"]
    else:
        prob = map_struct["prob"]

    ranked_tile_probs = compute_tiles_map(tile_struct, prob, func='np.sum(x)')
    ranked_tile_times = gwemopt.utils.integrationTime(tot_obs_time, ranked_tile_probs, func=None, T_int=config_struct["exposuretime"])

    if config_struct["FOV_type"] == "square":
        FOV = config_struct["FOV"]*config_struct["FOV"]
    elif config_struct["FOV_type"] == "circle":
        FOV = np.pi*config_struct["FOV"]*config_struct["FOV"]
    lim_mag = config_struct["magnitude"]
    lim_time = config_struct["exposuretime"]

    tau = np.arange(lim_time,3600.0,lim_time)
    Loftau = None

    N_ref = 9.7847e9
    L_min = 4.9370e31
    L_max = 4.9370e33

    if "distmu" in map_struct:
        prob = map_struct["prob"]
        distnorm = map_struct["distnorm"]
        distmu = map_struct["distmu"]
        distsigma = map_struct["distsigma"]

        D_min = 1.0e7
        D_max = 1.0e10
        R = np.linspace(D_min/1e6, D_max/1e6)
        p_R = [np.sum(prob * rr**2 * distnorm * norm(distmu, distsigma).pdf(rr)) for rr in R]
        p_R = np.array(p_R)
        p_R = p_R / len(prob)

        R = R*1e6
        D_mu = None
        D_sig = None
    else:
        D_mu = 200.0e6
        D_sig = 60.0e6
        R = None
        p_R = None

    tau, prob = gwemopt.pem.Pem(FOV, lim_mag, lim_time, N_ref = N_ref, L_min = L_min, L_max = L_max, tau = tau, Loftau = Loftau, D_mu = D_mu, D_sig = D_sig, R = R, p_R = p_R)

    tprob, time_allocation = gwemopt.pem.Main(tot_obs_time, 0, 0, ranked_tile_probs, tau, prob, 'Eq')

    if params["doPlots"]:
        gwemopt.plotting.tauprob(params,tau,prob)

    keys = tile_struct.keys()
    for key, prob, exposureTime in zip(keys, ranked_tile_probs, time_allocation):
        tile_struct[key]["prob"] = prob
        tile_struct[key]["exposureTime"] = exposureTime

    return tile_struct

def compute_tiles_map(tile_struct, skymap, func=None):

    if func is None:
        f = lambda x: np.sum(x)
    else:
        f = lambda x: eval(func)

    ntiles = len(tile_struct.keys())
    vals = np.nan*np.ones((ntiles,))
    for ii in tile_struct.iterkeys():
        vals[ii] = f(skymap[tile_struct[ii]["ipix"]])

    return vals
