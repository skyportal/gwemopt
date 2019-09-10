
import os, sys
import copy
import numpy as np
import healpy as hp
import gwemopt.plotting
from astropy.time import Time

import gwemopt.utils
import gwemopt.rankedTilesGenerator
import gwemopt.scheduler
import ligo.segments as segments

def combine_coverage_structs(coverage_structs):

    coverage_struct_combined = {}
    coverage_struct_combined["data"] = np.empty((0,8))
    coverage_struct_combined["filters"] = np.empty((0,1))
    coverage_struct_combined["ipix"] = []
    coverage_struct_combined["patch"] = []
    coverage_struct_combined["FOV"] = np.empty((0,1))
    coverage_struct_combined["area"] = np.empty((0,1))
    coverage_struct_combined["telescope"] = np.empty((0,1))
    coverage_struct_combined["galaxies"] = []    

    for coverage_struct in coverage_structs:
        coverage_struct_combined["data"] = np.append(coverage_struct_combined["data"],coverage_struct["data"],axis=0)
        coverage_struct_combined["filters"] = np.append(coverage_struct_combined["filters"],coverage_struct["filters"])
        coverage_struct_combined["ipix"] = coverage_struct_combined["ipix"] + coverage_struct["ipix"]
        coverage_struct_combined["patch"] = coverage_struct_combined["patch"] + coverage_struct["patch"]
        coverage_struct_combined["FOV"] = np.append(coverage_struct_combined["FOV"],coverage_struct["FOV"])
        coverage_struct_combined["area"] = np.append(coverage_struct_combined["area"],coverage_struct["area"])
        coverage_struct_combined["telescope"] = np.append(coverage_struct_combined["telescope"],coverage_struct["telescope"])
        if "galaxies" in coverage_struct:
            coverage_struct_combined["galaxies"] = coverage_struct_combined["galaxies"] + coverage_struct["galaxies"]

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
            raise ValueError("Change distance scale...")
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

    map_struct_hold = copy.deepcopy(map_struct)

    coverage_structs = []
    n_scope = 0
    full_prob_map = map_struct["prob"]

    for jj, telescope in enumerate(params["telescopes"]):

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

        # Try to load the minimum duration of time from telescope config file
        # Otherwise set it to zero
        try:   
            min_obs_duration = config_struct["min_observability_duration"] / 24
        except:
            min_obs_duration = 0.0

        if "filt_change_time" in config_struct.keys(): filt_change_time = config_struct["filt_change_time"]
        else: filt_change_time = 0

        if params["doIterativeTiling"] and (params["tilesType"] == "galaxy"):
            tile_struct = gwemopt.utils.slice_galaxy_tiles(params, tile_struct, combine_coverage_structs(coverage_structs))

        if params["doPerturbativeTiling"] and (jj>0) and (not params["tilesType"] == "galaxy"):
            tile_struct = gwemopt.utils.perturb_tiles(params, config_struct, telescope, map_struct_hold, tile_struct)
        
        if params["doOverlappingScheduling"]:
            tile_struct = gwemopt.utils.check_overlapping_tiles(params, tile_struct, combine_coverage_structs(coverage_structs))

        if params["doAlternatingFilters"]:
            params_hold = copy.copy(params)
            tile_struct_hold = copy.copy(tile_struct)
            config_struct_hold = copy.copy(config_struct)
            coverage_struct,tile_struct_hold = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold, telescope, map_struct_hold, tile_struct_hold)

            if params["doBalanceExposure"]:
                filters = params["filters"]
                prob={} #for reassigning probs in loop
                for key in tile_struct.keys():
                    prob[key] = tile_struct[key]['prob']

                keys_scheduled = coverage_struct["data"][:,5]
                filts_used = {key:[] for key in keys_scheduled}
                filts = coverage_struct["filters"]

                for (key,filt) in zip(keys_scheduled,filts): #finds # of tiles w/balanced exposures w/ no max tiles restriction
                    filts_used[key].append(filt)
                n_equal=0
                for key in filts_used:
                    if len(filts_used[key]) == len(filters):
                        n_equal+=1
                optimized_max=-1
                for key in tile_struct.keys(): #reassigns probabilities
                    tile_struct[key]['prob'] = prob[key]

                params["doMaxTiles"] = True
                countervals=[]

                max_trials = np.linspace(10,200,20) #runs through tiles in increments of 10

                for ii,max_trial in enumerate(max_trials):
                    params["max_nb_tiles"] = np.array([max_trial],dtype=np.float)
                    params_hold = copy.copy(params)
                    tile_struct_hold = copy.copy(tile_struct)
                    config_struct_hold = copy.copy(config_struct)
                    
                    coverage_struct_hold,tile_struct_hold = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold, telescope, map_struct_hold, tile_struct_hold)

                    keys_scheduled = coverage_struct_hold["data"][:,5]
                    filts_used = {key:[] for key in keys_scheduled}
                    filts = coverage_struct_hold["filters"]

                    for (key,filt) in zip(keys_scheduled,filts):
                        filts_used[key].append(filt)
                    counter=0
                    for key in filts_used:
                        if len(filts_used[key]) == len(filters):
                            counter+=1
                    countervals.append(counter)
                    if counter>=n_equal:
                        n_equal,optimized_max = counter,max_trial

                    for key in tile_struct.keys():
                        tile_struct[key]['prob'] = prob[key]
                    if ii>0 and counter<=countervals[ii-1]: break #breaks if # of fields w/ all exposures starts decreasing
                
                max_trials = np.linspace(optimized_max-9,optimized_max+9,10) #increments of 2, 10 tiles below and above optimized estimate
                countervals=[] #to compare # of fields w/ all exposures between different iterations
                
                for ii,max_trial in enumerate(max_trials):
                    if optimized_max==-1: break #breaks if no max tiles restriction should be imposed
                    params["max_nb_tiles"] = np.array([max_trial],dtype=np.float)
                    params_hold = copy.copy(params)
                    tile_struct_hold = copy.copy(tile_struct)
                    config_struct_hold = copy.copy(config_struct)
                    coverage_struct_hold,tile_struct_hold = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold, telescope, map_struct_hold, tile_struct_hold)

                    keys_scheduled = coverage_struct_hold["data"][:,5]
                    filts_used = {key:[] for key in keys_scheduled}
                    filts = coverage_struct_hold["filters"]

                    for (key,filt) in zip(keys_scheduled,filts):
                        filts_used[key].append(filt)
                    counter=0
                    for key in filts_used:
                        if len(filts_used[key]) == len(filters):
                            counter+=1
                    countervals.append(counter)
                    if counter>=n_equal:
                        n_equal,optimized_max = counter,max_trial

                    for key in tile_struct.keys():
                        tile_struct[key]['prob'] = prob[key]
                    if ii>0 and counter<=countervals[ii-1]: break #breaks if # of fields w/ all exposures starts decreasing
                
                optimized_max_prev=optimized_max
                #another for loop for more accurate estimation, iff optimized_max is below a certain number (to save runtime)
                if optimized_max<=50:
                    max_trials=np.linspace(optimized_max-1,optimized_max+1,3)
                    countervals=[]
                    for ii,max_trial in enumerate(max_trials):
                        if optimized_max==-1: break #breaks if no max tiles restriction should be imposed
                        if max_trial==optimized_max_prev: continue #doesn't redo maxtile # that has already been tested
                        params["max_nb_tiles"] = np.array([max_trial],dtype=np.float)
                        params_hold = copy.copy(params)
                        tile_struct_hold = copy.copy(tile_struct)
                        config_struct_hold = copy.copy(config_struct)
                        
                        coverage_struct_hold,tile_struct_hold = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold, telescope, map_struct_hold, tile_struct_hold)
                    
                        keys_scheduled = coverage_struct_hold["data"][:,5]
                        filts_used = {key:[] for key in keys_scheduled}
                        filts = coverage_struct_hold["filters"]
                    
                        for (key,filt) in zip(keys_scheduled,filts):
                            filts_used[key].append(filt)
                        counter=0
                        for key in filts_used:
                            if len(filts_used[key]) == len(filters):
                                counter+=1
                        countervals.append(counter)
                        if counter>=n_equal:
                            n_equal,optimized_max = counter,max_trial

                        for key in tile_struct.keys():
                            tile_struct[key]['prob'] = prob[key]
                        if ii>0 and counter<=countervals[ii-1]: break #breaks if # of fields w/ all exposures starts decreasing

                params["max_nb_tiles"] = np.array([optimized_max],dtype=np.float)

                params_hold = copy.copy(params)
                tile_struct_hold = copy.copy(tile_struct)
                config_struct_hold = copy.copy(config_struct)
                coverage_struct,tile_struct_hold = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold, telescope, map_struct_hold, tile_struct_hold)

                params_hold = copy.copy(params)
                config_struct_hold = copy.copy(config_struct)
                tile_struct_hold, doReschedule = gwemopt.utils.balance_tiles(params_hold, telescope, tile_struct_hold, coverage_struct)
                
                if doReschedule:
                    coverage_struct,tile_struct_hold = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold, telescope, map_struct_hold, tile_struct_hold)
                
                #raises exception if no fields w/ all requested filters are found
                keys_scheduled = coverage_struct["data"][:,5]
                filts = coverage_struct["filters"]
                filts_used = {key:[] for key in keys_scheduled}
                equal_filts = False
                for (key,filt) in zip(keys_scheduled,filts): #checks if there are any fields w/ obs. in all filters
                    filts_used[key].append(filt)
                    if len(filts_used[key]) == len(filters):
                        equal_filts=True
                        break
                if not equal_filts:
                    filters = ','.join(filters)
                    raise Exception(f'No fields were found with observations in filters {filters}')


        else:
            if not params["tilesType"] == "galaxy":
                tile_struct = gwemopt.tiles.powerlaw_tiles_struct(params, config_struct, telescope, map_struct_hold, tile_struct)      
            else:
                for key in tile_struct.keys():
                    # Check that a given tile is observable a minimum amount of time
                    # If not set the proba associated to the tile to zero
                    if 'segmentlist' and 'prob' in tile_struct[key] and tile_struct[key]['segmentlist'] and min_obs_duration > 0.0:
                        observability_duration = 0.0
                        for counter in range(len(tile_struct[key]['segmentlist'])):
                            observability_duration += tile_struct[key]['segmentlist'][counter][1] - tile_struct[key]['segmentlist'][counter][0]
                        if tile_struct[key]['prob'] > 0.0 and observability_duration < min_obs_duration:
                            tile_struct[key]['prob'] = 0.0
            
            if params["timeallocationType"] == "manual": #only works if using same telescope
                try:
                    for field_id in params["observedTiles"]:
                        field_id = int(field_id)
                        if field_id in tile_struct:
                            tile_struct[field_id]['prob'] = 0.0
            
                except:
                    raise ValueError("need to specify tiles that have been observed using --observedTiles")
                        
            if params["doSuperSched"]:
                tile_struct = erase_observed_tiles(params,tile_struct)
            
            coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)

            if params["doBalanceExposure"]:
                cnt,ntrials = 0,10
                while cnt < ntrials:
                    tile_struct, doReschedule = gwemopt.utils.balance_tiles(params, telescope, tile_struct, coverage_struct)
                    if doReschedule:
                        coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)
                        cnt = cnt+1
                    else: break

            if params["doMaxTiles"]:
                tile_struct, doReschedule = gwemopt.utils.slice_number_tiles(params, telescope, tile_struct, coverage_struct)    
                if doReschedule:
                    coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)

        tile_structs[telescope] = tile_struct
        coverage_structs.append(coverage_struct)

        if params["doIterativeTiling"]:
            map_struct_hold = gwemopt.utils.slice_map_tiles(params, map_struct_hold, coverage_struct)
               
    map_struct["prob"] = full_prob_map

    if params["doMovie_supersched"]:
        gwemopt.plotting.doMovie_supersched(params,combine_coverage_structs(coverage_structs),tile_structs,map_struct)
    
    return tile_structs, combine_coverage_structs(coverage_structs)

def pem(params, map_struct, tile_structs):

    map_struct_hold = copy.deepcopy(map_struct)

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
                tile_struct_hold = gwemopt.tiles.pem_tiles_struct(params, config_struct, telescope, map_struct_hold, tile_struct_hold)
                coverage_struct_hold = gwemopt.scheduler.scheduler(params, config_struct, tile_struct_hold)
                coverage_structs_hold.append(coverage_struct_hold)
            coverage_struct = combine_coverage_structs(coverage_structs_hold)
        else:
            tile_struct = gwemopt.tiles.pem_tiles_struct(params, config_struct, telescope, map_struct_hold, tile_struct)
            coverage_struct = gwemopt.scheduler.scheduler(params, config_struct, tile_struct)
        coverage_structs.append(coverage_struct)

        if params["doIterativeTiling"]:
            map_struct_hold = gwemopt.utils.slice_map_tiles(map_struct_hold, coverage_struct)

    return combine_coverage_structs(coverage_structs)

def erase_observed_tiles(params,tile_struct):
    Tobs = list(params["Tobs_all"])
    Tobs = np.linspace(Tobs[0],Tobs[1],params["Tobs_split"]+1)

    if f'{Tobs[0]:.2}_to_{Tobs[1]:.2}' in params["outputDir"]: #only proceeds if not first round of Tobs
        return tile_struct
    for ii,Tob in enumerate(Tobs[:-1]): #finds out which round function is being called in to figure out previous telescopes/coverage_structs
        if f'{Tobs[ii]:.2}_to_{Tobs[ii+1]:.2}_Tobs' in params["outputDir"]:
            break

    while ii>0: #loops through all previous coverage structs + covered field ids to set tile probabilities to 0
        ii-=1
        prevtelescopes = params["alltelescopes"][ii].split(",")
        coverage_struct = params["coverage_structs"][f'coverage_struct_{ii}']
        tile_struct_hold = gwemopt.utils.check_overlapping_tiles(params,tile_struct,coverage_struct)
        for prevtelescope in prevtelescopes:
            for key in tile_struct.keys():
                if 'epochs' in tile_struct_hold[key]:
                    epochs = tile_struct_hold[key]["epochs"]
                    for jj in range(len(epochs)):
                        field_id = epochs[jj,5]
                        telescope = tile_struct[key]["epochs_telescope"][jj]
                        if field_id in params["covered_field_ids"][prevtelescope] and telescope == prevtelescope: #makes sure field id obtained from check_overlapping_tiles is for the correct telescope
                            tile_struct[key]['prob'] = 0.0
                            break

    return tile_struct

def timeallocation(params, map_struct, tile_structs):

    if params["timeallocationType"] == "powerlaw":
        print("Generating powerlaw schedule...")
        tile_structs, coverage_struct = gwemopt.coverage.powerlaw(params, map_struct, tile_structs)
    elif params["timeallocationType"] == "waw":
        if params["do3D"]:
            print("Generating WAW schedule...")
            coverage_struct = gwemopt.coverage.waw(params, map_struct, tile_structs)
        else:
            raise ValueError("Need to enable --do3D for waw")

    elif params["timeallocationType"] == "manual":
        print("Generating manual schedule...")
        tile_structs, coverage_struct = gwemopt.coverage.powerlaw(params, map_struct, tile_structs)

    elif params["timeallocationType"] == "pem":
        print("Generating PEM schedule...")
        coverage_struct = gwemopt.coverage.pem(params, map_struct, tile_structs)

    return tile_structs, coverage_struct 
