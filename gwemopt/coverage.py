
import os, sys
import numpy as np
import healpy as hp

from astropy.time import Time

import glue.segments

import gwemopt.utils
import gwemopt.rankedTilesGenerator
import gwemopt.moc, gwemopt.pem

def combine_coverage_structs(coverage_structs):

    coverage_struct_combined = {}
    coverage_struct_combined["data"] = np.empty((0,5))
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
    coverage_struct["data"] = np.empty((0,5))
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

        coverage_struct["data"] = np.append(coverage_struct["data"],np.array([[ra,dec,mjd,mag,config_struct["exposuretime"]]]),axis=0)
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

def tiles_coverage(params, eventinfo, config_struct, tile_struct):

    nside = params["nside"]
    gpstime = eventinfo["gpstime"]
    mjd_inj = Time(gpstime, format='gps', scale='utc').mjd

    coverage_struct = {}
    coverage_struct["data"] = np.empty((0,5))
    coverage_struct["filters"] = []
    coverage_struct["ipix"] = []
    coverage_struct["patch"] = []
    coverage_struct["area"] = []

    segmentlist = glue.segments.segmentlist()
    n_windows = len(params["Tobs"]) // 2
    start_segments = mjd_inj + params["Tobs"][::2]
    end_segments = mjd_inj + params["Tobs"][1::2]
    for start_segment, end_segment in zip(start_segments,end_segments):
        segmentlist.append(glue.segments.segment(start_segment,end_segment))

    keys = tile_struct.keys()
    while len(keys) > 0 and len(segmentlist) > 0:
        key = keys[0]
        tile_struct_hold = tile_struct[key] 
        exposureTime = tile_struct_hold["exposureTime"]

        mjd_exposure_start = segmentlist[0][0]
        mjd_exposure_end = mjd_exposure_start + exposureTime/86400.0
        if mjd_exposure_end > segmentlist[0][1]:
            mjd_exposure_end = segmentlist[0][1]
            exposureTime = (mjd_exposure_end - mjd_exposure_start)*86400.0
            tile_struct[key]["exposureTime"] = tile_struct[key]["exposureTime"] - exposureTime 
        else:
            del tile_struct[key]
            keys.pop(0) 

        segment = glue.segments.segment(mjd_exposure_start,mjd_exposure_end)
        segmentlist = segmentlist - glue.segments.segmentlist([segment])
        segmentlist.coalesce()

        mjd_exposure_mid = (mjd_exposure_start+mjd_exposure_end)/2.0
        nexp = np.round(exposureTime/config_struct["exposuretime"])
        nmag = np.log(nexp) / np.log(2.5)
        mag = config_struct["magnitude"] + nmag

        coverage_struct["data"] = np.append(coverage_struct["data"],np.array([[tile_struct_hold["ra"],tile_struct_hold["dec"],mjd_exposure_mid,mag,exposureTime]]),axis=0)
        coverage_struct["filters"].append(config_struct["filter"])
        coverage_struct["patch"].append(tile_struct_hold["patch"])
        coverage_struct["ipix"].append(tile_struct_hold["ipix"])
        coverage_struct["area"].append(tile_struct_hold["area"])

    coverage_struct["area"] = np.array(coverage_struct["area"])
    coverage_struct["filters"] = np.array(coverage_struct["filters"])
    coverage_struct["FOV"] = config_struct["FOV"]*np.ones((len(coverage_struct["filters"]),))

    return coverage_struct

def waw(params, eventinfo, map_struct, tile_structs): 

    nside = params["nside"]

    t = np.arange(0,7,1/24.0)
    #t = np.arange(0,7,1.0)
    cr90 = map_struct["cumprob"] < 0.9
    detmaps = gwemopt.waw.detectability_maps(params, t, map_struct, verbose=True, limit_to_region=cr90, nside=nside)

    n_windows = len(params["Tobs"]) // 2
    tot_obs_time = np.sum(np.diff(params["Tobs"])[::2]) * 86400.

    coverage_structs = []
    for telescope in params["telescopes"]: 
        tile_struct = tile_structs[telescope]
        config_struct = params["config"][telescope]
        T_int = config_struct["exposuretime"]
        ranked_tile_probs = gwemopt.tiles.compute_tiles_map(tile_struct, map_struct["prob"], func='np.sum(x)')
        strategy_struct = gwemopt.waw.construct_followup_strategy_tiles(map_struct["prob"],detmaps,t,tile_struct,T_int,params["Tobs"])
        print np.sum(strategy_struct)
        strategy_struct = strategy_struct*86400.0

        if strategy_struct == None:
            print "Change distance scale..."
            exit(0)
        keys = tile_struct.keys()
        for key, prob, exposureTime in zip(keys, ranked_tile_probs, strategy_struct):
            tile_struct[key]["prob"] = prob
            tile_struct[key]["exposureTime"] = exposureTime

        coverage_struct = tiles_coverage(params, eventinfo, config_struct, tile_struct)

        coverage_structs.append(coverage_struct)

    if params["doPlots"]:
        gwemopt.plotting.waw(params,detmaps,t,strategy_struct)

    return combine_coverage_structs(coverage_structs)

def waterfall(params, eventinfo, map_struct, tile_structs):

    coverage_structs = []
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tile_struct = tile_structs[telescope]
        tile_struct = gwemopt.tiles.waterfall_tiles_struct(params, config_struct, telescope, map_struct, tile_struct)      
 
        coverage_struct = tiles_coverage(params, eventinfo, config_struct, tile_struct)
        coverage_structs.append(coverage_struct)

    return combine_coverage_structs(coverage_structs)

def pem(params, eventinfo, map_struct, tile_structs):

    n_windows = len(params["Tobs"]) // 2
    tot_obs_time = np.sum(np.diff(params["Tobs"])[::2]) * 86400

    coverage_structs = []
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tile_struct = tile_structs[telescope]
        tile_struct = gwemopt.tiles.pem_tiles_struct(params, config_struct, telescope, map_struct, tile_struct)

        coverage_struct = tiles_coverage(params, eventinfo, config_struct, tile_struct)
        coverage_structs.append(coverage_struct)

    return combine_coverage_structs(coverage_structs)

def summary(params, map_struct, eventinfo, coverage_struct):

    summaryfile = os.path.join(params["outputDir"],'summary.dat')
    fid = open(summaryfile,'w')

    gpstime = eventinfo["gpstime"]
    mjd_inj = Time(gpstime, format='gps', scale='utc').mjd

    tts = np.array([1,7,60])
    for tt in tts:

        radecs = np.empty((0,2))
        mjds_floor = []
        mjds = []
        ipixs = np.empty((0,2))
        cum_prob = 0.0
        cum_area = 0.0

        for ii in xrange(len(coverage_struct["ipix"])):
            data = coverage_struct["data"][ii,:]
            filt = coverage_struct["filters"][ii]
            ipix = coverage_struct["ipix"][ii]
            patch = coverage_struct["patch"][ii]
            FOV = coverage_struct["FOV"][ii]
            area = coverage_struct["area"][ii]

            prob = np.sum(map_struct["prob"][ipix])

            if data[2] > mjd_inj+tt:
                continue

            ipixs = np.append(ipixs,ipix)
            ipixs = np.unique(ipixs).astype(int)

            cum_prob = np.sum(map_struct["prob"][ipixs])
            cum_area = len(ipixs) * map_struct["pixarea_deg2"]
            mjds.append(data[2])
            mjds_floor.append(int(np.floor(data[2])))
        if len(mjds_floor) == 0:
            print "No images after %.1f days..."%tt
            fid.write('%.1f,-1,-1,-1,-1\n'%(tt))
        else: 

            mjds = np.unique(mjds)
            mjds_floor = np.unique(mjds_floor)

            print "After %.1f days..."%tt
            print "Number of hours after first image: %.5f"%(24*(np.min(mjds)-mjd_inj))
            print "MJDs covered: %s"%(" ".join(str(x) for x in mjds_floor))
            print "Cumultative probability: %.5f"%cum_prob
            print "Cumultative area: %.5f degrees"%cum_area

            fid.write('%.1f,%.5f,%.5f,%.5f,%s\n'%(tt,24*(np.min(mjds)-mjd_inj),cum_prob,cum_area," ".join(str(x) for x in mjds_floor)))

    fid.close()

