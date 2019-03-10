
import os, sys
import copy
import numpy as np
import healpy as hp

from astropy.time import Time

import gwemopt.utils
import gwemopt.rankedTilesGenerator
import gwemopt.scheduler

def combine_coverage_structs(coverage_structs):

    coverage_struct_combined = {}
    coverage_struct_combined["data"] = np.empty((0,8))
    coverage_struct_combined["filters"] = np.empty((0,1))
    coverage_struct_combined["ipix"] = []
    coverage_struct_combined["patch"] = []
    coverage_struct_combined["FOV"] = np.empty((0,1))
    coverage_struct_combined["area"] = np.empty((0,1))
    for coverage_struct in coverage_structs:
        coverage_struct_combined["data"] = np.append(coverage_struct_combined["data"],coverage_struct["data"],axis=0)
        coverage_struct_combined["filters"] = np.append(coverage_struct_combined["filters"],coverage_struct["filters"])
        coverage_struct_combined["ipix"] = coverage_struct_combined["ipix"] + coverage_struct["ipix"]
        coverage_struct_combined["patch"] = coverage_struct_combined["patch"] + coverage_struct["patch"]
        coverage_struct_combined["FOV"] = np.append(coverage_struct_combined["FOV"],coverage_struct["FOV"])
        coverage_struct_combined["area"] = np.append(coverage_struct_combined["area"],coverage_struct["area"])

    return coverage_struct_combined

def read_coverage(params, telescope, filename):

    nside = params["nside"]
    config_struct = params["config"][telescope]

    lines = [line.rstrip('\n') for line in open(filename)]
    lines = lines[1:]
    lines = filter(None,lines)

    coverage_struct = {}
    coverage_struct["data"] = np.empty((0,8))
    coverage_struct["filters"] = []
    coverage_struct["ipix"] = []
    coverage_struct["patch"] = []
    coverage_struct["area"] = []

    for line in lines:
        lineSplit = line.split(",")
        ra = float(lineSplit[2])
        dec = float(lineSplit[3])
        mjd = float(lineSplit[4])
        filt = lineSplit[6]
        mag = float(lineSplit[7])

        coverage_struct["data"] = np.append(coverage_struct["data"],np.array([[ra,dec,mjd,mag,config_struct["exposuretime"],-1,-1,-1]]),axis=0)
        coverage_struct["filters"].append(filt)

        if telescope == "ATLAS":
            alpha=0.2
            color='#6c71c4'
        elif telescope == "PS1":
            alpha=0.1
            color='#859900'
        else:
            alpha=0.2
            color='#6c71c4'

        if config_struct["FOV_coverage_type"] == "square":
            ipix, radecs, patch, area = gwemopt.utils.getSquarePixels(ra, dec, config_struct["FOV_coverage"], nside, alpha=alpha, color=color)
        elif config_struct["FOV_coverage_type"] == "circle":
            ipix, radecs, patch, area = gwemopt.utils.getCirclePixels(ra, dec, config_struct["FOV_coverage"], nside, alpha=alpha, color=color)

        coverage_struct["patch"].append(patch)
        coverage_struct["ipix"].append(ipix)
        coverage_struct["area"].append(area)

    coverage_struct["filters"] = np.array(coverage_struct["filters"])
    coverage_struct["area"] = np.array(coverage_struct["area"])
    coverage_struct["FOV"] = config_struct["FOV_coverage"]*np.ones((len(coverage_struct["filters"]),))

    return coverage_struct

def read_coverage_files(params):

    coverage_structs = []
    for telescope, coverageFile in zip(params["telescopes"],params["coverageFiles"]):
        coverage_struct = read_coverage(params,telescope,coverageFile)
        coverage_structs.append(coverage_struct)

    return combine_coverage_structs(coverage_structs)

def waw(params, map_struct, tile_structs): 

    nside = params["nside"]

    t = np.arange(0,7,1/24.0)
    #t = np.arange(0,7,1.0)
    cr90 = map_struct["cumprob"] < 0.9
    detmaps = gwemopt.waw.detectability_maps(params, t, map_struct, verbose=True, limit_to_region=cr90, nside=nside)

    coverage_structs = []
    for telescope in params["telescopes"]: 
        tile_struct = tile_structs[telescope]
        config_struct = params["config"][telescope]
        T_int = config_struct["exposuretime"]
        ranked_tile_probs = gwemopt.tiles.compute_tiles_map(tile_struct, map_struct["prob"], func='np.sum(x)')
        strategy_struct = gwemopt.waw.construct_followup_strategy_tiles(map_struct["prob"],detmaps,t,tile_struct,T_int,params["Tobs"])
        if strategy_struct is None:
            print("Change distance scale...")
            exit(0)
        print(np.sum(strategy_struct))
        strategy_struct = strategy_struct*86400.0
        keys = tile_struct.keys()
        for key, prob, exposureTime in zip(keys, ranked_tile_probs, strategy_struct):
            tile_struct[key]["prob"] = prob
            tile_struct[key]["exposureTime"] = exposureTime
            tile_struct[key]["nexposures"] = int(np.floor(exposureTime/config_struct["exposuretime"]))
        coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)
        coverage_structs.append(coverage_struct)

    if params["doPlots"]:
        gwemopt.plotting.waw(params,detmaps,t,strategy_struct)

    return combine_coverage_structs(coverage_structs)

def powerlaw(params, map_struct, tile_structs):

    coverage_structs = []
    n_scope = 0
    full_prob_map = map_struct["prob"]

    for telescope in params["telescopes"]:

        if params["doSplit"]:
            if "observability" in map_struct:
                map_struct["observability"][telescope]["prob"] = map_struct["groups"][n_scope]
            else:
                map_struct["prob"] = map_struct["groups"][n_scope]
            if n_scope < len(map_struct["groups"]) - 1:
                n_scope += 1
            else:
                n_scope = 0

        config_struct = params["config"][telescope]
        tile_struct = tile_structs[telescope]

        if params["doAlternatingFilters"]:
            filters, exposuretimes = params["filters"], params["exposuretimes"]
            tile_struct_hold = copy.copy(tile_struct)
            coverage_structs_hold = []
            cnt = 0
            for filt, exposuretime in zip(filters,exposuretimes):
                params["filters"] = [filt]
                params["exposuretimes"] = [exposuretime]
                
                if cnt > 0:
                   if len(coverage_struct_hold["exposureused"]) > 0:
                       maxidx = (np.max(coverage_struct_hold["exposureused"])+1).astype(int)
                       if maxidx > len(config_struct["exposurelist"]):
                           exposurelistnew = config_struct["exposurelist"][maxidx:]
                           config_struct["exposurelist"] = exposurelistnew

                tile_struct_hold = gwemopt.tiles.powerlaw_tiles_struct(params, config_struct, telescope, map_struct, tile_struct_hold)      
                coverage_struct_hold = gwemopt.scheduler.scheduler(params, config_struct, tile_struct_hold)
                coverage_structs_hold.append(coverage_struct_hold)
                cnt = cnt + 1
            coverage_struct = combine_coverage_structs(coverage_structs_hold)
        else:
            tile_struct = gwemopt.tiles.powerlaw_tiles_struct(params, config_struct, telescope, map_struct, tile_struct)      
            coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)
        coverage_structs.append(coverage_struct)

    map_struct["prob"] = full_prob_map
    return combine_coverage_structs(coverage_structs)

def pem(params, map_struct, tile_structs):

    coverage_structs = []
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tile_struct = tile_structs[telescope]

        if params["doAlternatingFilters"]:
            filters, exposuretimes = params["filters"], params["exposuretimes"]
            tile_struct_hold = copy.copy(tile_struct)
            coverage_structs_hold = []
            for filt, exposuretime in zip(filters,exposuretimes):
                params["filters"] = [filt]
                params["exposuretimes"] = [exposuretime]
                tile_struct_hold = gwemopt.tiles.pem_tiles_struct(params, config_struct, telescope, map_struct, tile_struct_hold)
                coverage_struct_hold = gwemopt.scheduler.scheduler(params, config_struct, tile_struct_hold)
                coverage_structs_hold.append(coverage_struct_hold)
            coverage_struct = combine_coverage_structs(coverage_structs_hold)
        else:
            tile_struct = gwemopt.tiles.pem_tiles_struct(params, config_struct, telescope, map_struct, tile_struct)
            coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)
        coverage_structs.append(coverage_struct)

    return combine_coverage_structs(coverage_structs)

def timeallocation(params, map_struct, tile_structs):

    if params["timeallocationType"] == "powerlaw":
        print("Generating powerlaw schedule...")
        coverage_struct = gwemopt.coverage.powerlaw(params, map_struct, tile_structs)
    elif params["timeallocationType"] == "waw":
        if params["do3D"]:
            print("Generating WAW schedule...")
            coverage_struct = gwemopt.coverage.waw(params, map_struct, tile_structs)
        else:
            print("Need to enable --do3D for waw")
            exit(0)
    elif params["timeallocationType"] == "pem":
        print("Generating PEM schedule...")
        coverage_struct = gwemopt.coverage.pem(params, map_struct, tile_structs)

    return coverage_struct 
