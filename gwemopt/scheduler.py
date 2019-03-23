
import os, sys, copy
import numpy as np
import healpy as hp

from VOEventLib.VOEvent import Table, Field, What
from VOEventLib.Vutil import utilityTable, stringVOEvent, VOEventExportClass

import astropy.coordinates
from astropy.time import Time, TimeDelta
import astropy.units as u
import ligo.segments as segments
import gwemopt.utils
import gwemopt.rankedTilesGenerator
import gwemopt.hungarian

def get_altaz_tiles(ras, decs, observatory, obstime):

    # Convert to RA, Dec.
    radecs = astropy.coordinates.SkyCoord(
            ra=np.array(ras)*u.degree, dec=np.array(decs)*u.degree, frame='icrs')

    # Alt/az reference frame at observatory, now
    frame = astropy.coordinates.AltAz(obstime=obstime, location=observatory)
            
    # Transform grid to alt/az coordinates at observatory, now
    altaz = radecs.transform_to(frame)    

    return altaz


def find_tile(exposureids_tile,exposureids,probs, idxs = None, exptimecheckkeys = []):
    # exposureids_tile: {expo id}-> list of the tiles available for observation
    # exposureids: list of tile ids for every exposure it is allocated to observe
    if idxs is not None:
        for idx in idxs:
            if len(exposureids_tile["exposureids"])-1 < idx: continue
            idx2 = exposureids_tile["exposureids"][idx]
            if idx2 in exposureids and not idx2 in exptimecheckkeys:
                idx = exposureids.index(idx2)
                exposureids.pop(idx)
                probs.pop(idx)
                return idx2, exposureids, probs

    findTile = True
    while findTile:
        if not exposureids_tile["probs"]:
            idx2 = -1
            findTile = False
            break
        idx = np.argmax(exposureids_tile["probs"])
        idx2 = exposureids_tile["exposureids"][idx]

        if exposureids:
            if idx2 in exposureids and not idx2 in exptimecheckkeys:
                idx = exposureids.index(idx2)
                exposureids.pop(idx)
                probs.pop(idx)
                findTile = False
            else:
                exposureids_tile["exposureids"].pop(idx)
                exposureids_tile["probs"].pop(idx)
        else:
            findTile = False

    return idx2, exposureids, probs


def find_tile_greedy_slew(current_time, current_ra, current_dec, tilesegmentlists, tileprobs, config_struct, tile_struct, 
        keynames, tileAllocatedTime, exptimecheckkeys = [], idle = 0):
    next_obs = -1
    idx2 = -1
    score_selected = -1
    slew_readout_selected = -1
    explength_selected = -1
    for ii in range(len(tilesegmentlists)): # for every tile
        key = keynames[ii]
        # exclude some tiles
        if keynames[ii] in exptimecheckkeys or np.absolute(tileAllocatedTime[ii]) < 1e-5:
            continue
        # calculate slew readout time
        distance = np.sqrt((tile_struct[key]['ra'] - current_ra)**2 +  (tile_struct[key]['dec'] - current_dec)**2)
        slew_readout = np.max([config_struct['readout'], distance / config_struct['slew_rate']])
        slew_readout = np.max([slew_readout - idle, 0])
        slew_readout = slew_readout / 86400
        for jj in range(len(tilesegmentlists[ii])): # for every segment
            seg = tilesegmentlists[ii][jj]
            if current_time + slew_readout < seg[1]: # if ends later than current + slew
                if current_time + slew_readout >= seg[0]: # if starts earlier than current + slew
                    # calculate the score
                    explength = np.min([seg[1] - current_time - slew_readout, tileAllocatedTime[ii]])
                    score = tileprobs[ii] * explength / (1 + slew_readout)
                    if idx2 == -1 or score > score_selected:
                        idx2 = keynames[ii]
                        score_selected = score
                        slew_readout_selected = slew_readout
                        explength_selected = explength
                elif idx2 == -1: # if starts later than current + slew
                    if next_obs == -1 or next_obs > seg[0]:
                        next_obs = seg[0]
                break
    exp_idle_seg = None
    if idx2 != -1:
        exp_idle_seg = segments.segment(current_time + slew_readout, current_time + slew_readout + explength_selected)
    elif next_obs != -1:
        exp_idle_seg = segments.segment(current_time, next_obs)
    return idx2, slew_readout_selected, exp_idle_seg


def get_order(params, tile_struct, tilesegmentlists, exposurelist, observatory, config_struct):
    '''
    tile_struct: dictionary. key -> struct info. 
    tilesegmentlists: list of lists. Segments for each tile in tile_struct 
        that are available for observation.
    exposurelist: list of segments that the telescope is supposed to be working.
        consecutive segments from the start to the end, with each segment size
        being the exposure time.
    Returns a list of tile indices in the order of observation.
    '''
    keys = tile_struct.keys()

    exposureids_tiles = {}
    first_exposure = np.inf*np.ones((len(keys),))
    last_exposure = -np.inf*np.ones((len(keys),))
    tileprobs = np.zeros((len(keys),))
    tilenexps = np.zeros((len(keys),))
    tileexptime = np.zeros((len(keys),))
    tilefilts = {}
    tileavailable = np.zeros((len(keys),))
    tileavailable_tiles = {} 
    keynames = []

    nexps = 0
    for jj, key in enumerate(keys):
        tileprobs[jj] = tile_struct[key]["prob"]
        tilenexps[jj] = tile_struct[key]["nexposures"]
        tilefilts[key] = copy.deepcopy(tile_struct[key]["filt"])
        tileavailable_tiles[jj] = []
        keynames.append(key) 

        nexps = nexps + tile_struct[key]["nexposures"]

    if "dec_constraint" in config_struct:
        dec_constraint = config_struct["dec_constraint"].split(",")
        dec_min = float(dec_constraint[0])
        dec_max = float(dec_constraint[1])

    for ii in range(len(exposurelist)):
        exposureids_tiles[ii] = {}
        exposureids = []
        probs = []
        for jj, key in enumerate(keys):
            tilesegmentlist = tilesegmentlists[jj]
            if tilesegmentlist.intersects_segment(exposurelist[ii]):
                if tile_struct[key]["prob"] == 0: continue
                if "dec_constraint" in config_struct:
                    print(tile_struct[key]["dec"],dec_min,dec_max)
                    if (tile_struct[key]["dec"] < dec_min) or (tile_struct[key]["dec"] > dec_max): continue
                exposureids.append(key)
                probs.append(tile_struct[key]["prob"])

                first_exposure[jj] = np.min([first_exposure[jj],ii])
                last_exposure[jj] = np.max([last_exposure[jj],ii])
                tileavailable_tiles[jj].append(ii)
                tileavailable[jj] = tileavailable[jj] + 1
        # in every exposure, the tiles available for observation
        exposureids_tiles[ii]["exposureids"] = exposureids # list of tile ids
        exposureids_tiles[ii]["probs"] = probs # the corresponding probs

    exposureids = []
    probs = []
    ras, decs = [], []
    for ii, key in enumerate(keys):
        # tile_struct[key]["nexposures"]: the number of exposures assigned to this tile
        for jj in range(tile_struct[key]["nexposures"]): 
            exposureids.append(key) # list of tile ids for every exposure it is allocated to observe
            probs.append(tile_struct[key]["prob"])
            ras.append(tile_struct[key]["ra"])
            decs.append(tile_struct[key]["dec"])

    idxs = -1*np.ones((len(exposureids_tiles.keys()),))
    filts = ['n'] * len(exposureids_tiles.keys())

    if nexps == 0:
        return idxs, filts    

    if params["scheduleType"] == "airmass_weighted":

        # # first step is to sort the array in order of descending probability
        # indsort = np.argsort(-np.array(probs))
        # probs = np.array(probs)[indsort]
        # ras = np.array(ras)[indsort]
        # decs = np.array(decs)[indsort]
        # exposureids = np.array(exposureids)[indsort] 

        matrix = []
        probmatrix = []

        old_exposureids = exposureids.copy()
        for ii in np.arange(len(exposurelist)):

            # first, check whether the entire patch is above the horizon
            t = Time(exposurelist[ii][0], format='mjd')
            altaz = get_altaz_tiles(ras, decs, observatory, t)
            alts = altaz.alt.degree
            horizon = config_struct["horizon"]
            horizon_mask = alts <= horizon
            airmass = 1 / np.cos((90. - alts) * np.pi / 180.)
            below_horizon_mask = horizon_mask * 10.**100
            airmass = airmass + below_horizon_mask

            # if part of patch below horizon, apply greedy method
            # temporary solution until runtime issue fixed
            if horizon_mask.any() or len(exposureids)==1:
                exptimecheck = np.where(exposurelist[ii][0]-tileexptime <
                    params["mindiff"]/86400.0)[0]
                exptimecheckkeys = [keynames[x] for x in exptimecheck]
                # find_tile finds the tile that covers the largest probablity
                # restricted by availability of tile and timeallocation
                idx2, exposureids, probs = find_tile(exposureids_tiles[ii],exposureids,probs,exptimecheckkeys=exptimecheckkeys)
                
                if idx2 in old_exposureids:
                    idx = old_exposureids.index(idx2)
                    ras.pop(idx)
                    decs.pop(idx)
                    old_exposureids.pop(idx)

                if idx2 in keynames:
                    idx = keynames.index(idx2)
                    tilenexps[idx] = tilenexps[idx] - 1
                    tileexptime[idx] = exposurelist[ii][0]
                    if len(tilefilts[idx2]) > 0:
                        filt = tilefilts[idx2].pop(0)
                        filts[ii] = filt

                idxs[ii] = idx2

            # once the patch has risen, create a matrix of the remaining exposure times and tiles
            elif not horizon_mask.any() and len(exposureids) > 1:
                matrix.append(probs/airmass)
                probmatrix.append(probs * (True^horizon_mask))

            tilematrix = np.array(matrix)

    if params["scheduleType"] == "greedy":
        for ii in np.arange(len(exposurelist)): 
            exptimecheck = np.where(exposurelist[ii][0]-tileexptime <
                                    params["mindiff"]/86400.0)[0]
            exptimecheckkeys = [keynames[x] for x in exptimecheck]

            # find_tile finds the tile that covers the largest probablity
            # restricted by availability of tile and timeallocation
            idx2, exposureids, probs = find_tile(exposureids_tiles[ii],exposureids,probs,exptimecheckkeys=exptimecheckkeys)
            if idx2 in keynames:
                idx = keynames.index(idx2)
                tilenexps[idx] = tilenexps[idx] - 1
                tileexptime[idx] = exposurelist[ii][0]
                if len(tilefilts[idx2]) > 0:
                    filt = tilefilts[idx2].pop(0)
                    filts[ii] = filt
            idxs[ii] = idx2

            if not exposureids: break

    elif params["scheduleType"] == "sear":
        #for ii in np.arange(len(exposurelist)):
        iis = np.arange(len(exposurelist)).tolist()
        while len(iis) > 0: 
            ii = iis[0]
            mask = np.where((ii == last_exposure) & (tilenexps > 0))[0]

            exptimecheck = np.where(exposurelist[ii][0]-tileexptime <
                                    params["mindiff"]/86400.0)[0]
            exptimecheckkeys = [keynames[x] for x in exptimecheck]

            if len(mask) > 0:
                idxsort = mask[np.argsort(tileprobs[mask])]
                idx2, exposureids, probs = find_tile(exposureids_tiles[ii],exposureids,probs,idxs=idxsort,exptimecheckkeys=exptimecheckkeys) 
                last_exposure[mask] = last_exposure[mask] + 1
            else:
                idx2, exposureids, probs = find_tile(exposureids_tiles[ii],exposureids,probs,exptimecheckkeys=exptimecheckkeys)
            if idx2 in keynames:
                idx = keynames.index(idx2)
                tilenexps[idx] = tilenexps[idx] - 1
                tileexptime[idx] = exposurelist[ii][0]
                if len(tilefilts[idx2]) > 0:
                    filt = tilefilts[idx2].pop(0)
                    filts[ii] = filt
            idxs[ii] = idx2 
            iis.pop(0)

            if not exposureids: break

    elif params["scheduleType"] == "weighted":
        for ii in np.arange(len(exposurelist)):
            jj = exposureids_tiles[ii]["exposureids"]
            weights = tileprobs[jj] * tilenexps[jj] / tileavailable[jj]
            weights[~np.isfinite(weights)] = 0.0

            exptimecheck = np.where(exposurelist[ii][0]-tileexptime <
                                    params["mindiff"]/86400.0)[0]
            weights[exptimecheck] = 0.0

            if np.any(weights >= 0):
                idxmax = np.argmax(weights)
                idx2 = jj[idxmax]
                if idx2 in keynames:
                    idx = keynames.index(idx2)
                    tilenexps[idx] = tilenexps[idx] - 1
                    tileexptime[idx] = exposurelist[ii][0]
                    if len(tilefilts[idx2]) > 0:
                        filt = tilefilts[idx2].pop(0)
                        filts[ii] = filt
                idxs[ii] = idx2
            tileavailable[jj] = tileavailable[jj] - 1  

    elif params["scheduleType"] == "airmass_weighted":
        
        tilematrix_mask = tilematrix > 10**(-10)

        if tilematrix_mask.any():
            hungarian = gwemopt.hungarian.Hungarian(tilematrix, profit_matrix=True)
            hungarian.calculate()
            optimal_points = np.array(hungarian.assigned_points)
            # sort the optimal points first on probability
            order = np.argsort(optimal_points[:, 0])
            optimal_points = optimal_points[order]
            max_no_observ = min(tilematrix.shape)

            for jj in range(max_no_observ):
                # idx0 indexes over the time windows, idx1 indexes over the probabilities
                # idx2 gets the exposure id of the tile, used to assign tileexptime and tilenexps
                try:
                    idx0 = optimal_points[jj][1]
                    idx1 = optimal_points[jj][0]
                    idx2 = exposureids[idx1.astype(int)]
                    tileprob = probmatrix[idx0][idx1]
                    if len(tilefilts[idx2]) > 0:
                        filt = tilefilts[idx2].pop(0)
                        filts[jj] = filt
                except: continue

                greedy_tile_idx = len(exposurelist) - max(tilematrix.shape)
                idxs[idx0 + greedy_tile_idx] = idx2


        else: print("The localization is not visible from the site.")

    else:
        print("Scheduling options are greedy/sear/weighted/airmass_weighted, or with _slew.")
        exit(0)

    return idxs, filts


def get_order_slew(params, tile_struct, tilesegmentlists, config_struct):   
    keys = tile_struct.keys() 
    keynames = []
    namekeys = {}
    tilefilts = {}
    tileAllocatedTime = {}
    tileprobs = np.zeros((len(keys),))
    for jj, key in enumerate(keys):
        tileprobs[jj] = tile_struct[key]["prob"]
        tilefilts[key] = copy.deepcopy(tile_struct[key]["filt"])
        keynames.append(key) 
        namekeys[key] = jj
        tileAllocatedTime[key] = tile_struct[key]["exposureTime"] / 86400
    lastObs = np.zeros((len(keys),))

    idxs = []
    filts = []
    exposurelist = []

    if params["scheduleType"] == "greedy_slew":
        current_time = Time(params['gpstime'], format='gps', scale='utc').mjd + np.min(params["Tobs"][::2])
        current_ra = config_struct["latitude"]
        current_dec = config_struct["longitude"]
        idle = 0
        while True:
            # check the time gap since last observation
            exptimecheck = np.where(current_time - lastObs < params["mindiff"]/86400.0)[0]
            exptimecheckkeys = [keynames[x] for x in exptimecheck]
            idx2, slew_readout, exp_idle_seg = find_tile_greedy_slew(current_time, current_ra, current_dec, tilesegmentlists, tileprobs, 
                    config_struct, tile_struct, keynames, tileAllocatedTime, exptimecheckkeys=exptimecheckkeys, idle=idle)
            if idx2 != -1:
                exp_idle_len = exp_idle_seg[1] - exp_idle_seg[0]
                idle = 0
                idxs.append(idx2)
                exposurelist.append(exp_idle_seg)
                filts.append(tilefilts[idx2].pop(0))
                tileAllocatedTime[idx2] = tileAllocatedTime[idx2] - exp_idle_len
                current_time = exp_idle_seg[1]
                current_ra = tile_struct[idx2]["ra"]
                current_dec = tile_struct[idx2]["dec"]
                idx = namekeys[idx2]
                lastObs[idx] = current_time
            elif exp_idle_seg is not None:
                exp_idle_len = exp_idle_seg[1] - exp_idle_seg[0]
                idxs.append(idx2)
                exposurelist.append(exp_idle_seg)
                filts.append('n')
                idle = exp_idle_len
                current_time += exp_idle_len
            else:
                break
    elif params["scheduleType"] == "sear_slew":
        print('sear_slew is not ready yet.')
    elif params["scheduleType"] == "weighted_slew":
        print('weighted_slew is not ready yet.')
    else:
        print("Scheduling options are greedy/sear/weighted, or with _slew.")
        exit(0)

    return idxs, exposurelist, filts


def scheduler(params, config_struct, tile_struct):
    '''
    config_struct: the telescope configurations
    tile_struct: the tiles, contains time allocation information
    '''
    import gwemopt.segments
    #import gwemopt.segments_astroplan
    coverage_struct = {}
    coverage_struct["data"] = np.empty((0,8))
    coverage_struct["filters"] = []
    coverage_struct["ipix"] = []
    coverage_struct["patch"] = []
    coverage_struct["area"] = []

    observatory = astropy.coordinates.EarthLocation(
        lat=config_struct["latitude"]*u.deg, lon=config_struct["longitude"]*u.deg, height=config_struct["elevation"]*u.m)

    segmentlist = config_struct["segmentlist"]
    exposurelist = config_struct["exposurelist"]
    #tilesegmentlists = gwemopt.segments_astroplan.get_segments_tiles(config_struct, tile_struct, observatory, segmentlist)
    tilesegmentlists = []
    keys = tile_struct.keys()
    for key in keys:
        # segments.py: tile_struct[key]["segmentlist"] is a list of segments when the tile is available for observation
        tilesegmentlists.append(tile_struct[key]["segmentlist"]) 
    print("Generating schedule order...")
    if params["scheduleType"].endswith('_slew'):
        keys, exposurelist, filts = get_order_slew(params, tile_struct, tilesegmentlists, config_struct)
    else:
        keys, filts = get_order(params,tile_struct,tilesegmentlists,exposurelist,observatory, config_struct)
    if params["doPlots"]:
        gwemopt.plotting.scheduler(params,exposurelist,keys)

    exposureused = np.where(np.array(keys)>=0)[0]
    coverage_struct["exposureused"] = exposureused
    while len(exposurelist) > 0:
        key, filt = keys[0], filts[0]
        if key == -1:
            keys = keys[1:]
            filts = filts[1:]
            exposurelist = exposurelist[1:]
        else:
            tile_struct_hold = tile_struct[key]
            mjd_exposure_start = exposurelist[0][0]
            nkeys = len(keys)
            for jj in range(nkeys):
                if (keys[jj] == key) and (filts[jj] == filt) and not (nkeys == jj+1):
                    mjd_exposure_end = exposurelist[jj][1]
                elif (keys[jj] == key) and (filts[jj] == filt) and (nkeys == jj+1):
                    mjd_exposure_end = exposurelist[jj][1]
                    nexp = jj + 1
                    keys = []
                    filts = []
                    exposurelist = []
                else:
                    nexp = jj + 1
                    keys = keys[jj:]
                    filts = filts[jj:]
                    exposurelist = exposurelist[jj:]
                    break    
 
            # the (middle) tile observation time is mjd_exposure_mid
            mjd_exposure_mid = (mjd_exposure_start+mjd_exposure_end)/2.0

            # calculate airmass for each tile at the middle of its exposure:
            t = Time(mjd_exposure_mid, format='mjd')
            altaz = get_altaz_tiles(tile_struct_hold["ra"], tile_struct_hold["dec"], observatory, t)
            alt = altaz.alt.degree
            airmass = 1 / np.cos((90. - alt) * np.pi / 180)

            nmag = np.log(nexp) / np.log(2.5)
            mag = config_struct["magnitude"] + nmag
            exposureTime = (mjd_exposure_end-mjd_exposure_start)*86400.0

            coverage_struct["data"] = np.append(coverage_struct["data"],np.array([[tile_struct_hold["ra"],tile_struct_hold["dec"],mjd_exposure_mid,mag,exposureTime,int(key),tile_struct_hold["prob"],airmass]]),axis=0)

            coverage_struct["filters"].append(filt)
            coverage_struct["patch"].append(tile_struct_hold["patch"])
            coverage_struct["ipix"].append(tile_struct_hold["ipix"])
            coverage_struct["area"].append(tile_struct_hold["area"])

    coverage_struct["area"] = np.array(coverage_struct["area"])
    coverage_struct["filters"] = np.array(coverage_struct["filters"])
    coverage_struct["FOV"] = config_struct["FOV"]*np.ones((len(coverage_struct["filters"]),))
    return coverage_struct


def computeSlewReadoutTime(config_struct, coverage_struct):
    slew_rate = config_struct['slew_rate']
    readout = config_struct['readout']
    prev_ra = config_struct["latitude"]
    prev_dec = config_struct["longitude"]
    acc_time = 0
    for dat in coverage_struct['data']:
        slew_readout_time = np.maximum(np.sqrt((prev_ra - dat[0])**2 + (prev_dec - dat[1])**2) / slew_rate, readout)
        acc_time += slew_readout_time
        prev_dec = dat[0]
        prev_ra = dat[1]
    return acc_time

def write_xml(xmlfile,map_struct,coverage_struct,config_struct):

    what = What()

    table = Table(name="data", Description=["The datas of GWAlert"])
    table.add_Field(Field(name=r"grid_id", ucd="", unit="", dataType="int", \
                    Description=["ID of the grid of fov"]))
    table.add_Field(Field(name="field_id", ucd="", unit="", dataType="int",\
                    Description=["ID of the filed"]))
    table.add_Field(
        Field(
            name=r"ra", ucd=r"pos.eq.ra ", unit="deg", dataType="float",
            Description=["The right ascension at center of fov in equatorial coordinates"]
            )
        )
    table.add_Field(
        Field(
            name="dec", ucd="pos.eq.dec ", unit="deg", dataType="float",
            Description=["The declination at center of fov in equatorial coordinates"]
            )
        )
    table.add_Field(
        Field(
            name="ra_width", ucd=" ", unit="deg", dataType="float",
            Description=["Width in RA of the fov"]
            )
        )
    table.add_Field(
        Field(
            name="dec_width", ucd="", unit="deg", dataType="float",
            Description=["Width in Dec of the fov"]
            )
        )
    table.add_Field(
        Field(
            name="prob_sum", ucd="", unit="None", dataType="float",
            Description=["The sum of all pixels in the fov"]
            )
        )
    table.add_Field(
        Field(
            name="observ_time", ucd="", unit="sec", dataType="float",
            Description=["Tile mid. observation time in MJD"]
            )
        )
    table.add_Field(
        Field(
            name="airmass", ucd="", unit="None", dataType="float",
            Description=["Airmass of tile at mid. observation time"]
            )
        )
    table.add_Field(Field(name="priority", ucd="", unit="", dataType="int", Description=[""]))
    table_field = utilityTable(table)
    table_field.blankTable(len(coverage_struct))

    for ii in range(len(coverage_struct["ipix"])):
        data = coverage_struct["data"][ii,:]
        filt = coverage_struct["filters"][ii]
        ipix = coverage_struct["ipix"][ii]
        patch = coverage_struct["patch"][ii]
        FOV = coverage_struct["FOV"][ii]
        area = coverage_struct["area"][ii]

        prob = np.sum(map_struct["prob"][ipix])

        ra, dec = data[0], data[1]
        observ_time, exposure_time, field_id, prob, airmass = data[2], data[4], data[5], data[6], data[7]

        table_field.setValue("grid_id", ii, 0)
        table_field.setValue("field_id", ii, field_id)
        table_field.setValue("ra", ii, ra)
        table_field.setValue("dec", ii, dec)
        table_field.setValue("ra_width", ii, config_struct["FOV"])
        table_field.setValue("dec_width", ii, config_struct["FOV"])
        table_field.setValue("observ_time", ii, observ_time)
        table_field.setValue("airmass", ii, airmass)
        table_field.setValue("prob_sum", ii, prob)
        table_field.setValue("priority", ii, ii)
    table = table_field.getTable()
    what.add_Table(table)
    xml = stringVOEvent(what)
    lines = xml.splitlines()
    linesrep = []
    for line in lines:
        linenew = line.replace(">b'",">").replace("'</","</").replace("=b'","=").replace("'>",">")
        linesrep.append(linenew)
    xmlnew = "\n".join(linesrep)
    fid = open(xmlfile, "w")
    fid.write(xmlnew)
    fid.close()

def summary(params, map_struct, coverage_struct):

    idx50 = len(map_struct["cumprob"])-np.argmin(np.abs(map_struct["cumprob"]-0.50))
    idx90 = len(map_struct["cumprob"])-np.argmin(np.abs(map_struct["cumprob"]-0.90))

    mapfile = os.path.join(params["outputDir"],'map.dat')
    fid = open(mapfile,'w')
    fid.write('%.5f %.5f\n'%(map_struct["pixarea_deg2"]*idx50,map_struct["pixarea_deg2"]*idx90))
    fid.close()
    filts = list(set(coverage_struct["filters"]))
    for telescope in params["telescopes"]:

        schedulefile = os.path.join(params["outputDir"],'schedule_%s.dat'%telescope)
        schedulexmlfile = os.path.join(params["outputDir"],'schedule_%s.xml'%telescope)
        config_struct = params["config"][telescope]

        write_xml(schedulexmlfile,map_struct,coverage_struct,config_struct)

        if (params["tilesType"] == "hierarchical") or (params["tilesType"] == "greedy"):
            fields = np.zeros((params["Ntiles"],len(filts)+2))
        else:
            fields = np.zeros((len(config_struct["tesselation"]),len(filts)+2))

        fid = open(schedulefile,'w')
        for ii in range(len(coverage_struct["ipix"])):
            data = coverage_struct["data"][ii,:]
            filt = coverage_struct["filters"][ii]
            ipix = coverage_struct["ipix"][ii]
            patch = coverage_struct["patch"][ii]
            FOV = coverage_struct["FOV"][ii]
            area = coverage_struct["area"][ii]

            prob = np.sum(map_struct["prob"][ipix])

            ra, dec = data[0], data[1]
            observ_time, exposure_time, field_id, prob, airmass = data[2], data[4], data[5], data[6], data[7]
            fid.write('%d %.5f %.5f %.5f %d %.5f %.5f %s\n'%(field_id,ra,dec,observ_time,exposure_time,prob,airmass,filt))

            idx1 = np.argmin(np.sqrt((config_struct["tesselation"][:,1]-data[0])**2 + (config_struct["tesselation"][:,2]-data[1])**2))
            idx2 = filts.index(filt)
            fields[idx1,0] = config_struct["tesselation"][idx1,0]
            fields[idx1,1] = prob
            fields[idx1,idx2+2] = fields[idx1,idx2+2]+1
        fid.close()

        idx = np.where(fields[:,1]>0)[0]
        fields = fields[idx,:]
        idx = np.argsort(fields[:,1])[::-1]
        fields = fields[idx,:]

        fields_sum = np.sum(fields[:,2:],axis=1)
        idx = np.where(fields_sum >= 2)[0]
        print('%d/%d fields were observed at least twice\n'%(len(idx),len(fields_sum)))

        print('Integrated probability, All: %.5f, 2+: %.5f'%(np.sum(fields[:,1]),np.sum(fields[idx,1])))

        slew_readout_time = computeSlewReadoutTime(config_struct, coverage_struct)
        print('Expected time spent on slewing and readout ' + str(slew_readout_time) + ' s.')

        coveragefile = os.path.join(params["outputDir"],'coverage_%s.dat'%telescope)
        fid = open(coveragefile,'w')
        for field in fields:
            fid.write('%d %.10f '%(field[0],field[1]))
            for ii in range(len(filts)):
                 fid.write('%d '%(field[2+ii]))
            fid.write('\n')
        fid.close()

    summaryfile = os.path.join(params["outputDir"],'summary.dat')
    fid = open(summaryfile,'w')

    gpstime = params["gpstime"]
    event_mjd = Time(gpstime, format='gps', scale='utc').mjd

    tts = np.array([1,7,60])
    for tt in tts:

        radecs = np.empty((0,2))
        mjds_floor = []
        mjds = []
        ipixs = np.empty((0,2))
        cum_prob = 0.0
        cum_area = 0.0

        for ii in range(len(coverage_struct["ipix"])):
            data = coverage_struct["data"][ii,:]
            filt = coverage_struct["filters"][ii]
            ipix = coverage_struct["ipix"][ii]
            patch = coverage_struct["patch"][ii]
            FOV = coverage_struct["FOV"][ii]
            area = coverage_struct["area"][ii]

            prob = np.sum(map_struct["prob"][ipix])

            if data[2] > event_mjd+tt:
                continue

            ipixs = np.append(ipixs,ipix)
            ipixs = np.unique(ipixs).astype(int)

            cum_prob = np.sum(map_struct["prob"][ipixs])
            cum_area = len(ipixs) * map_struct["pixarea_deg2"]
            mjds.append(data[2])
            mjds_floor.append(int(np.floor(data[2])))

        if len(mjds_floor) == 0:
            print("No images after %.1f days..."%tt)
            fid.write('%.1f,-1,-1,-1,-1\n'%(tt))
        else:

            mjds = np.unique(mjds)
            mjds_floor = np.unique(mjds_floor)

            print("After %.1f days..."%tt)
            print("Number of hours after first image: %.5f"%(24*(np.min(mjds)-event_mjd)))
            print("MJDs covered: %s"%(" ".join(str(x) for x in mjds_floor)))
            print("Cumultative probability: %.5f"%cum_prob)
            print("Cumultative area: %.5f degrees"%cum_area)

            fid.write('%.1f,%.5f,%.5f,%.5f,%s\n'%(tt,24*(np.min(mjds)-event_mjd),cum_prob,cum_area," ".join(str(x) for x in mjds_floor)))

    fid.close()

