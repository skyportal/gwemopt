
import os, sys
import copy

import numpy as np
import healpy as hp

from scipy.stats import norm

from astropy.time import Time

from shapely.geometry import MultiPoint

import gwemopt.utils
import gwemopt.rankedTilesGenerator
import gwemopt.samplers, gwemopt.segments
import gwemopt.quadrants
import gwemopt.moc

from gwemopt.segments import angular_distance

def get_rectangle(ras, decs, ra_size, dec_size):

    ras[ras>180.0] = ras[ras>180.0] - 360.0

    poly = MultiPoint([(x,y) for x,y in zip(ras,decs)]).envelope
    minx, miny, maxx, maxy = poly.bounds
    width = maxx - minx
    height = maxy - miny

    while (width < ra_size) or (height < dec_size):

        ra_mean, dec_mean = np.mean(ras), np.mean(decs)    
        dist = angular_distance(ra_mean, dec_mean,
                                ras, decs)
        idx = np.setdiff1d(np.arange(len(ras)),np.argmax(dist))
        ras, decs = ras[idx], decs[idx]

        if len(ras) == 1:
            return np.mod(ras[0], 360.0), decs[0]

        poly = MultiPoint([(x,y) for x,y in zip(ras,decs)]).envelope
        minx, miny, maxx, maxy = poly.bounds
        width = maxx - minx
        height = maxy - miny

    return np.mod((minx+maxx)/2.0, 360.0), (miny+maxy)/2.0    

def galaxy(params, map_struct, catalog_struct):
    nside = params["nside"]

    tile_structs = {}
    for telescope in params["telescopes"]:

        config_struct = params["config"][telescope]

        # Combine in a single pointing, galaxies that are distant by
        # less than FoV * params['galaxies_FoV_sep']
        # Take galaxy with highest proba at the center of new pointing
        FoV = params["config"][telescope]['FOV'] * params['galaxies_FoV_sep']
        new_ra = []
        new_dec = []
        new_Sloc = []
        new_S = []
        galaxies = []
        idxRem = np.arange(len(catalog_struct["ra"])).astype(int)

        while len(idxRem) > 0:
            ii = idxRem[0]
            ra, dec, Sloc, S = catalog_struct["ra"][ii], catalog_struct["dec"][ii], catalog_struct["Sloc"][ii], catalog_struct["S"][ii]    
            
            if config_struct["FOV_type"] == "square":
                decCorners = (dec - FoV, dec + FoV)
                # assume small enough to use average dec for corners
                raCorners = (ra - FoV/np.cos(np.deg2rad(dec)) , ra + FoV / np.cos(np.deg2rad(dec)))
                idx1 = np.where((catalog_struct["ra"][idxRem]>=raCorners[0]) & (catalog_struct["ra"][idxRem]<=raCorners[1]))[0]
                idx2 = np.where((catalog_struct["dec"][idxRem]>=decCorners[0]) & (catalog_struct["dec"][idxRem]<=decCorners[1]))[0]
                mask = np.intersect1d(idx1,idx2)

                if len(mask) > 1:
                    ra_center, dec_center = get_rectangle(catalog_struct["ra"][idxRem][mask], catalog_struct["dec"][idxRem][mask], FoV/np.cos(np.deg2rad(dec)), FoV)

                    decCorners = (dec_center - FoV/2.0, dec_center + FoV/2.0)
                    raCorners = (ra_center - FoV/(2.0*np.cos(np.deg2rad(dec))) , ra_center + FoV/(2.0*np.cos(np.deg2rad(dec))))
                    idx1 = np.where((catalog_struct["ra"][idxRem]>=raCorners[0]) & (catalog_struct["ra"][idxRem]<=raCorners[1]))[0]
                    idx2 = np.where((catalog_struct["dec"][idxRem]>=decCorners[0]) & (catalog_struct["dec"][idxRem]<=decCorners[1]))[0]
                    mask2 = np.intersect1d(idx1,idx2)
                    # did the optimization help?
                    if len(mask2) > 2:
                        mask = mask2
                else:
                    ra_center, dec_center = np.mean(catalog_struct["ra"][idxRem][mask]), np.mean(catalog_struct["dec"][idxRem][mask])

            elif config_struct["FOV_type"] == "circle":
                dist = angular_distance(ra, dec,
                                        catalog_struct["ra"][idxRem],
                                        catalog_struct["dec"][idxRem])
                mask = np.where((2 * FoV) >= dist)[0]
                if len(mask) > 1:
                    ra_center, dec_center = get_rectangle(catalog_struct["ra"][idxRem][mask], catalog_struct["dec"][idxRem][mask], (FoV/np.sqrt(2))/np.cos(np.deg2rad(dec)), FoV/np.sqrt(2))

                    dist = angular_distance(ra_center, dec_center,
                                            catalog_struct["ra"][idxRem],
                                            catalog_struct["dec"][idxRem])
                    mask2 = np.where(FoV >= dist)[0]
                    # did the optimization help?
                    if len(mask2) > 2:
                        mask = mask2
                else:
                    ra_center, dec_center = np.mean(catalog_struct["ra"][idxRem][mask]), np.mean(catalog_struct["dec"][idxRem][mask])

            new_ra.append(ra_center)
            new_dec.append(dec_center)
            new_Sloc.append(np.sum(catalog_struct["Sloc"][idxRem][mask]))
            new_S.append(np.sum(catalog_struct["S"][idxRem][mask]))
            galaxies.append(idxRem[mask])

            idxRem = np.setdiff1d(idxRem, idxRem[mask])

        # redefine catalog_struct
        catalog_struct_new = {}
        catalog_struct_new["ra"] = new_ra
        catalog_struct_new["dec"] = new_dec
        catalog_struct_new["Sloc"] = new_Sloc
        catalog_struct_new["S"] = new_S
        catalog_struct_new["galaxies"] = galaxies

        moc_struct = {}
        cnt = 0
        for ra, dec, Sloc, S in zip(catalog_struct_new["ra"], catalog_struct_new["dec"], catalog_struct_new["Sloc"], catalog_struct_new["S"]):
            moc_struct[cnt] = gwemopt.moc.Fov2Moc(params, config_struct, telescope, ra, dec, nside)
            cnt = cnt + 1
        
        tile_struct = powerlaw_tiles_struct(params, config_struct, telescope, map_struct, moc_struct)
        tile_struct = gwemopt.segments.get_segments_tiles(params, config_struct, tile_struct)

        cnt = 0
        for ra, dec, Sloc, S, galaxies in zip(catalog_struct_new["ra"], catalog_struct_new["dec"], catalog_struct_new["Sloc"], catalog_struct_new["S"],catalog_struct_new["galaxies"]):
            if params["galaxy_grade"] == "Sloc":
                tile_struct[cnt]['prob'] = Sloc
            elif params["galaxy_grade"] == "S":
                tile_struct[cnt]['prob'] = S

            tile_struct[cnt]['galaxies'] = galaxies
            if config_struct["FOV_type"] == "square":
                tile_struct[cnt]['area'] = params["config"][telescope]['FOV']**2
            elif config_struct["FOV_type"] == "circle":
                tile_struct[cnt]['area'] = 4*np.pi*params["config"][telescope]['FOV']**2
            cnt = cnt + 1

        tile_structs[telescope] = tile_struct

    return tile_structs

def greedy(params, map_struct):

    tile_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]

        tile_struct = gwemopt.samplers.greedy_tiles_struct(params, config_struct, telescope, map_struct)
        tile_struct = gwemopt.segments.get_segments_tiles(params, config_struct, tile_struct)

        tile_structs[telescope] = tile_struct

    return tile_structs

def hierarchical(params, map_struct):

    tile_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]

        tile_struct = gwemopt.samplers.hierarchical_tiles_struct(params, config_struct, telescope, map_struct)
        tile_struct = gwemopt.segments.get_segments_tiles(params, config_struct, tile_struct)

        tile_structs[telescope] = tile_struct

    return tile_structs

def powerlaw_tiles_struct(params, config_struct, telescope, map_struct, tile_struct):

    keys = tile_struct.keys()
    ntiles = len(keys)
    if ntiles == 0:
        return tile_struct

    tot_obs_time = config_struct["tot_obs_time"]

    if "observability" in map_struct:
        prob = map_struct["observability"][telescope]["prob"]
    else:
        prob = map_struct["prob"]

    n, cl, dist_exp = params["powerlaw_n"], params["powerlaw_cl"], params["powerlaw_dist_exp"]
    
    if params["tilesType"] == "galaxy":
        tile_probs = compute_tiles_map(params, tile_struct, prob, func='center', ipix_keep=map_struct["ipix_keep"])
    else:
        tile_probs = compute_tiles_map(params, tile_struct, prob, func='np.sum(x)', ipix_keep=map_struct["ipix_keep"])

    tile_probs[tile_probs<np.max(tile_probs)*0.01] = 0.0
 
    prob_scaled = copy.deepcopy(prob)
    prob_sorted = np.sort(prob_scaled)[::-1]
    prob_indexes = np.argsort(prob_scaled)[::-1]
    prob_cumsum = np.cumsum(prob_sorted)
    index = np.argmin(np.abs(prob_cumsum - cl)) + 1
    #prob_scaled[prob_indexes[index:]] = 1e-10
    prob_scaled[prob_indexes[index:]] = 0.0
    prob_scaled = prob_scaled**n
    prob_scaled = prob_scaled / np.nansum(prob_scaled)
   
    if params["tilesType"] == "galaxy":
        ranked_tile_probs = compute_tiles_map(params, tile_struct, prob_scaled, func='center', ipix_keep=map_struct["ipix_keep"])
    else:
        ranked_tile_probs = compute_tiles_map(params, tile_struct, prob_scaled, func='np.sum(x)', ipix_keep=map_struct["ipix_keep"])

    ranked_tile_probs[np.isnan(ranked_tile_probs)] = 0.0
    ranked_tile_probs_thresh = np.max(ranked_tile_probs)*0.01
    ranked_tile_probs[ranked_tile_probs<=ranked_tile_probs_thresh] = 0.0
    ranked_tile_probs = ranked_tile_probs / np.nansum(ranked_tile_probs)

    if "distmed" in map_struct:
        distmed = map_struct["distmed"]
        distmed[distmed<=0] = np.nan
        distmed[~np.isfinite(distmed)] = np.nan
        #distmed[distmed<np.nanmedian(distmed)/4.0] = np.nanmedian(distmed)/4.0

        ranked_tile_distances = compute_tiles_map(params, tile_struct, distmed, func='np.nanmedian(x)')        
        ranked_tile_distances_median = ranked_tile_distances / np.nanmedian(ranked_tile_distances)
        ranked_tile_distances_median = ranked_tile_distances_median**dist_exp
        ranked_tile_probs = ranked_tile_probs*ranked_tile_distances_median
        ranked_tile_probs = ranked_tile_probs / np.nansum(ranked_tile_probs)
        ranked_tile_probs[np.isnan(ranked_tile_probs)] = 0.0
        
    if params["doSingleExposure"]:
        keys = tile_struct.keys()
        ranked_tile_times = np.zeros((len(ranked_tile_probs),len(params["exposuretimes"])))
        for ii in range(len(params["exposuretimes"])):
            ranked_tile_times[ranked_tile_probs>0,ii] = params["exposuretimes"][ii]
        for key, prob, exposureTime, tileprob in zip(keys, ranked_tile_probs, ranked_tile_times, tile_probs):

            # Try to load the minimum duration of time from telescope config file
            # Otherwise set it to zero
            try:
                min_obs_duration = config_struct["min_observability_duration"] / 24
            except:
                min_obs_duration = 0.0

            # Check that a given tile is observable a minimum amount of time
            # If not set the proba associated to the tile to zero
            if 'segmentlist' and 'prob' in tile_struct[key] and tile_struct[key]['segmentlist'] and min_obs_duration > 0.0:
                observability_duration = 0.0 
                for counter in range(len(tile_struct[key]['segmentlist'])):
                    observability_duration += tile_struct[key]['segmentlist'][counter][1] - tile_struct[key]['segmentlist'][counter][0]
                if tile_struct[key]['prob'] > 0.0 and observability_duration < min_obs_duration: 
                   tileprob = 0.0


            tile_struct[key]["prob"] = tileprob
            if prob == 0.0:
                tile_struct[key]["exposureTime"] = 0.0
                tile_struct[key]["nexposures"] = 0
                tile_struct[key]["filt"] = []
            else:
                if params["doReferences"] and (telescope in ["ZTF", "DECam"]):
                    tile_struct[key]["exposureTime"] = []
                    tile_struct[key]["nexposures"] = []
                    tile_struct[key]["filt"] = []
                    if key in config_struct["reference_images"]:
                        for ii in range(len(params["filters"])):
                            if params["filters"][ii] in config_struct["reference_images"][key]:
                                tile_struct[key]["exposureTime"].append(exposureTime[ii])
                                tile_struct[key]["filt"].append(params["filters"][ii])
                        tile_struct[key]["nexposures"] = len(tile_struct[key]["exposureTime"])
                    else:
                        tile_struct[key]["exposureTime"] = 0.0
                        tile_struct[key]["nexposures"] = 0
                        tile_struct[key]["filt"] = []
                else:
                    tile_struct[key]["exposureTime"] = exposureTime
                    tile_struct[key]["nexposures"] = len(params["exposuretimes"])
                    tile_struct[key]["filt"] = params["filters"]

    else:
        ranked_tile_times = gwemopt.utils.integrationTime(tot_obs_time, ranked_tile_probs, func=None, T_int=config_struct["exposuretime"])

        keys = tile_struct.keys()

        for key, prob, exposureTime, tileprob in zip(keys, ranked_tile_probs, ranked_tile_times, tile_probs):
            # Try to load the minimum duration of time from telescope config file
            # Otherwise set it to zero
            try:
                min_obs_duration = config_struct["min_observability_duration"] / 24
            except:
                min_obs_duration = 0.0

            # Check that a given tile is observable a minimum amount of time
            # If not set the proba associated to the tile to zero
            if 'segmentlist' and 'prob' in tile_struct[key] and tile_struct[key]['segmentlist'] and min_obs_duration > 0.0:
                observability_duration = 0.0 
                for counter in range(len(tile_struct[key]['segmentlist'])):
                    observability_duration += tile_struct[key]['segmentlist'][counter][1] - tile_struct[key]['segmentlist'][counter][0]
                if tile_struct[key]['prob'] > 0.0 and observability_duration < min_obs_duration: 
                    prob = 0.0

            tile_struct[key]["prob"] = prob
            tile_struct[key]["exposureTime"] = exposureTime
            tile_struct[key]["nexposures"] = int(np.floor(exposureTime/config_struct["exposuretime"]))
            tile_struct[key]["filt"] = [config_struct["filt"]] * tile_struct[key]["nexposures"]
            
    return tile_struct

def moc(params, map_struct, moc_structs):

    tile_structs = {}
    for telescope in params["telescopes"]:
 
        config_struct = params["config"][telescope]
        moc_struct = moc_structs[telescope]

        tile_struct = powerlaw_tiles_struct(params, config_struct, telescope, map_struct, moc_struct)
        tile_struct = gwemopt.segments.get_segments_tiles(params, config_struct, tile_struct)
        tile_structs[telescope] = tile_struct

    return tile_structs

def pem_tiles_struct(params, config_struct, telescope, map_struct, tile_struct):

    tot_obs_time = config_struct["tot_obs_time"]

    if "observability" in map_struct:
        prob = map_struct["observability"][telescope]["prob"]
    else:
        prob = map_struct["prob"]

    ranked_tile_probs = compute_tiles_map(params, tile_struct, prob, func='np.sum(x)', ipix_keep=map_struct["ipix_keep"])
    ranked_tile_times = gwemopt.utils.integrationTime(tot_obs_time, ranked_tile_probs, func=None, T_int=config_struct["exposuretime"])

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

    tau, prob = gwemopt.pem.Pem(lim_mag, lim_time, N_ref = N_ref, L_min = L_min, L_max = L_max, tau = tau, Loftau = Loftau, D_mu = D_mu, D_sig = D_sig, R = R, p_R = p_R)

    tprob, time_allocation = gwemopt.pem.Main(tot_obs_time, 0, 0, ranked_tile_probs, tau, prob, 'Eq')

    if params["doPlots"]:
        gwemopt.plotting.tauprob(params,tau,prob)

    keys = tile_struct.keys()
    for key, prob, exposureTime in zip(keys, ranked_tile_probs, time_allocation):
        tile_struct[key]["prob"] = prob
        tile_struct[key]["exposureTime"] = exposureTime
        tile_struct[key]["nexposures"] = int(np.floor(exposureTime/config_struct["exposuretime"]))
        tile_struct[key]["filt"] = [config_struct["filt"]] * tile_struct[key]["nexposures"]

    return tile_struct

def compute_tiles_map(params, tile_struct, skymap, func=None, ipix_keep=[]):

    if func is None:
        f = lambda x: np.sum(x)
        
    elif func == 'center':
        keys = tile_struct.keys()
        ntiles = len(keys)
        vals = np.nan*np.ones((ntiles,))
        nside = hp.npix2nside(len(skymap))
        for ii,key in enumerate(tile_struct.keys()):
            pix_center = hp.ang2pix(nside, tile_struct[key]['ra'], tile_struct[key]['dec'], lonlat=True)
            val = skymap[pix_center]
            vals[ii] = val        
        return vals
    
    else:
        f = lambda x: eval(func)

    prob = copy.deepcopy(skymap)

    keys = tile_struct.keys()
    ntiles = len(keys)
    vals = np.nan*np.ones((ntiles,))
    for ii,key in enumerate(tile_struct.keys()):
        idx = np.where(prob[tile_struct[key]["ipix"]] == -1)[0]
        idx = np.setdiff1d(idx,ipix_keep)
        if len(prob[tile_struct[key]["ipix"]]) == 0:
            rat = 0.0
        else:
            rat = float(len(idx)) / float(len(prob[tile_struct[key]["ipix"]]))
        if rat > params["maximumOverlap"]:
            vals[ii] = 0.0            
        else:
            if len(prob[tile_struct[key]["ipix"]]) == 0:
                vals[ii] = 0.0
            else:
                vals_to_sum = prob[tile_struct[key]["ipix"]]
                vals_to_sum[vals_to_sum < 0] = 0
                vals[ii] = f(vals_to_sum)
        ipix_slice = np.setdiff1d(tile_struct[key]["ipix"],ipix_keep)
        if len(ipix_slice) > 0:
            prob[ipix_slice] = -1

    return vals

def tesselation_spiral(config_struct, scale=0.80):
    if config_struct["FOV_type"] == "square":
        FOV = config_struct["FOV"]*config_struct["FOV"]*scale
    elif config_struct["FOV_type"] == "circle":
        FOV = np.pi*config_struct["FOV"]*config_struct["FOV"]*scale

    area_of_sphere = 4*np.pi*(180/np.pi)**2
    n = int(np.ceil(area_of_sphere/FOV))
    print("Using %d points to tile the sphere..."%n)
 
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)
 
    points = np.zeros((n, 3))
    points[:,0] = radius * np.cos(theta)
    points[:,1] = radius * np.sin(theta)
    points[:,2] = z

    ra, dec = hp.pixelfunc.vec2ang(points, lonlat=True)
    fid = open(config_struct["tesselationFile"],'w')
    for ii in range(len(ra)):
        fid.write('%d %.5f %.5f\n'%(ii,ra[ii],dec[ii]))
    fid.close()   

def tesselation_packing(config_struct, scale=0.97):
    sphere_radius = 1.0
    if config_struct["FOV_type"] == "square":
        circle_radius = np.deg2rad(config_struct["FOV"]/2.0) * scale
    elif config_struct["FOV_type"] == "circle":
        circle_radius = np.deg2rad(config_struct["FOV"]) * scale
    vertical_count = int( (np.pi*sphere_radius)/(2*circle_radius) )

    phis = []
    thetas = []

    phi = -0.5*np.pi
    phi_step = np.pi/vertical_count
    while phi<0.5*np.pi:
        horizontal_count = int( (2*np.pi*np.cos(phi)*sphere_radius)/(2*circle_radius) )
        if horizontal_count==0: horizontal_count=1
        theta = 0
        theta_step = 2*np.pi/horizontal_count
        while theta<2*np.pi-1e-8:
            phis.append(phi)
            thetas.append(theta)
            theta += theta_step
        phi += phi_step
    dec = np.array(np.rad2deg(phis))
    ra = np.array(np.rad2deg(thetas))

    fid = open(config_struct["tesselationFile"],'w')
    for ii in range(len(ra)):
        fid.write('%d %.5f %.5f\n'%(ii,ra[ii],dec[ii]))
    fid.close()

