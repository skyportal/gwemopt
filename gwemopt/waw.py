# WAW strategy implemented here: https://github.com/omsharansalafia/waw
# Original by Om Sharan Salafia
# Modified by Michael Coughlin

from __future__ import print_function

import numpy as np
import healpy as hp

from scipy.stats import norm

import gwemopt.lightcurve

def detectability_maps(params, t, map_struct, samples_struct=None, nside=256, verbose=False, limit_to_region=None):
    """
    Compute the detectability maps P(F(t)>Flim|ra,dec,...) for a given EM
    counterpart, at a given obs frequency, with a given flux limit, at
    given times after the merger.


    Parameters:
    -----------
    nu_obs: scalar
        Observing frequency in Hz.
    Flim: scalar
        Limiting flux density of the search in mJy.
    t: numpy 1D array
        The times [in days] at which the detectability maps must be computed.
    samples_struct: dictionary
        Dictionary containing the parameters of the posterior samples. The
        dictionary is passed to the counterpart function that computes/retrieves
        the lightcurves. The dictionary *must* contain at least the keys 'ra'
        and 'dec' that contain the sky positions of the posterior samples.
    counterpart: function
        The function that is used to compute/retrieve the counterpart lightcurves.
        It must comply with the input/output specifications defined in
        emcounterparts.IO_specifications.
    nside: int
        The nside parameter that defines the healpix resolution of the output
        detectability maps.
    verbose: boolean
        If True, print the advancement status of the lightcurve computation/retrieval.
    limit_to_region: numpy 1D array
        A healpix map of booleans. The detectability maps will be computed only
        at sky positions that correspond to True pixels in this map. If None,
        the computation will be all-sky.

    Returns:
    --------
    detmaps: ndarray
        The shape of the output ndarray is (nt,npix), where nt is the length
        of the input t array, and npix = 12*nside**2. In practice, detmaps[i]
        is the detectability map at time t[i]. Conversely, detmaps[:,p] is
        the time-dependent detectability at sky position p (p is the healpix
        index! The actual coordinates are given by hp.pix2ang(nside,p)).
    """

    lightcurve_structs = gwemopt.lightcurve.read_files(params["lightcurveFiles"])
    for key in lightcurve_structs.iterkeys():
        lightcurve_struct = lightcurve_structs[key]   

    # compute the lightcurve of the counterpart for each posterior sample
    #F = gwemopt.lightcurve.compute_apparent_magnitude_samples(params, lightcurve_struct, samples_struct, t)
    F = gwemopt.lightcurve.compute_apparent_magnitude(params, lightcurve_struct, t)
    Flim = params["config"][params["telescopes"][0]]["magnitude"] 

    # compute the detectability maps
    detmaps = np.empty([len(t), hp.nside2npix(nside)])

    if verbose:
        print("")
    for i in range(len(t)):
        if verbose:
            print(" #### computing detmap {0:d} of {1:d} ({2:.1f} percent completed) ####".format(i + 1, len(t),i / len(t) * 100.),end="\r")

        #detmaps[i] = sky_pos_cond_prob_gt(F[:, i], Flim, samples_struct['ra'], samples_struct['dec'], nside,limit_to_region)
        #detmaps[i] = sky_pos_cond_prob_gt(10**(F[:, i]/-2.5), 10**(Flim/-2.5), samples_struct['ra'], samples_struct['dec'], nside,limit_to_region)
        detmaps[i] = sky_pos_cond_prob(F[i], Flim, map_struct, nside, limit_to_region)

    if verbose:
        print(" #### {} detmaps computed (100 percent completed) ####".format(len(t)))

    return detmaps

def construct_followup_strategy_tiles(skymap, detmaps, t_detmaps, tile_struct, T_int, T_available, min_detectability=0.01):

    # make sure that the detectability is above the minimum at some point
    if np.all(detmaps[np.isfinite(detmaps)] < min_detectability):
        print("No point in the detmaps is above the minumum required detectability.")
        return None

    T = T_int
 
    ntiles = len(tile_struct.keys())
    ridx = np.arange(ntiles) # this keeps track of the original healpix indices corresponding to the region
    dm = np.empty([len(detmaps), ntiles])
    for i in range(len(detmaps)):
        dm[i] = gwemopt.tiles.compute_tiles_map(tile_struct, detmaps[i], func='np.mean(x)') 

    # bring the skymap to the same resolution, and take only the region
    sm = gwemopt.tiles.compute_tiles_map(tile_struct, skymap, func='np.sum(x)')
    # also, find the descending probability sorted indices of the skymap
    descending_prob_idx = np.argsort(sm)[::-1]

    # how much total observing time is available? Just sum the differences
    # between ending and starting times of the available time windows,
    # then convert to seconds
    n_windows = len(T_available) // 2
    tot_obs_time = np.sum(np.diff(T_available)[::2]) * 86400.
    # compute the number of time slots, and their starting times in days.
    # Note that the number of slots per window must be an integer, thus
    # a the end of each window there might be some remainder time unused
    slots_in_window = (np.diff(T_available)[::2] * 86400./T).astype(int)
    n_slots = np.sum(slots_in_window)
    t0_slot = np.empty(n_slots)
    k = 0
    for i in range(n_windows):
        for j in range(slots_in_window[i]):
            t0_slot[k] = T_available[2 * i] + j * T/86400.
            k = k + 1
    # mark all slots as available
    available_slots = np.ones(n_slots,dtype=bool)

    # the strategy will be a healpix map of observation times
    strategy = np.ones(len(sm))*np.nan

    # assign the available time slots to the skymap pixels in order
    # of descending sky position probability. Each pixel is assigned
    # the available time slot where the detectability is highest
    for p in descending_prob_idx:
        try: # find the time of best detectability
            detp = np.interp(t0_slot[available_slots],t_detmaps,dm[:,p])
            i_best = np.argmax(detp)
            i0_best = np.arange(len(available_slots))[available_slots][i_best] #original index of the best time slot
        except:
            continue

        # do not assign observation if the best detectability is below the requested limit
        if detp[i_best]<min_detectability:
            continue
        else:
            strategy[ridx[p]]=t0_slot[i0_best] # assign the observation time
            available_slots[i0_best]=False # mark the assigned slot as not available anymore

    strategy[np.isnan(strategy)] = 0.0
    return strategy
    #return np.ma.masked_invalid(strategy)

def construct_followup_strategy(skymap, detmaps, t_detmaps, Afov, T_int, T_available, min_detectability=0.01, limit_to_region=None):
    """
    Construct the EM follow-up as in Salafia+17.

    Parameters:
    -----------
    skymap: 1D numpy array
        The healpix map of GW posterior sky position probability per unit area.
    detmaps: ndarray
        The detectability maps as computed by ::function::detectability_maps.
    t_detmaps: 1D array
        The times (in days) corresponding to the detectability maps provided.
    Afov: scalar
        The field of view of the observing instrument in square degrees.
    T_int: scalar
        The exposure/integration time (in seconds) that corresponds to the
        search limiting flux.
    T_available: tuple
        The starting and ending times (in days after the merger) of the
        available time windows. Must be an even number! (each window has
        a starting time and an ending time).
    min_detectability: scalar
        The minimum value of the detectability for which an observation
        can be scheduled.
    ...

    Return:
    -------
    strategy: 1D masked numpy array
        A healpix map containing the observation times corresponding
        to the follow-up strategy

    """

    # make sure that the detectability is above the minimum at some point
    if np.all(detmaps[np.isfinite(detmaps)] < min_detectability):
        print("No point in the detmaps is above the minumum required detectability.")
        return None

    # determine the nside that makes the pixel area closest to the fov
    # area, but not larger (FIXME: a MOC based representation of the fov
    # would do much better!)
    nsides = 2. ** np.arange(2, 10)
    areas = hp.nside2pixarea(nsides, degrees=True)
    ratio = Afov / areas
    nside = np.min(nsides[ratio > 1.])

    # rescale the integration time to the effective one corresponding to
    # that pixel area
    Apixel = hp.nside2pixarea(nside, degrees=True)
    T = T_int * Apixel / Afov  # this is the time needed to cover one healpix pixel

    # bring the limit_to_region to the correct resolution
    if limit_to_region is None:
        region = np.ones(hp.nside2npix(nside),dtype=bool)
    else:
        #region = hp.ud_grade(limit_to_region, nside)
        limit_to_region_ones = np.zeros(len(limit_to_region))
        limit_to_region_ones[np.where(limit_to_region==True)[0]] = 1.0
        region = hp.ud_grade(limit_to_region_ones,nside)
        region = np.ceil(region)
        region = region.astype(bool)

    # limit the detmaps to the region (to save memory and computation time)
    # after degrading/upgrading to the correct resolution
    ridx = np.arange(hp.nside2npix(nside))[region] # this keeps track of the original healpix indices corresponding to the region

    dm = np.empty([len(detmaps), len(region[region])])

    for i in range(len(detmaps)):
        dm[i] = hp.ud_grade(detmaps[i], nside)[region]

    # bring the skymap to the same resolution, and take only the region
    sm = hp.ud_grade(skymap,nside)[region]
    # also, find the descending probability sorted indices of the skymap
    descending_prob_idx = np.argsort(sm)[::-1]

    # how much total observing time is available? Just sum the differences
    # between ending and starting times of the available time windows,
    # then convert to seconds
    n_windows = len(T_available) // 2
    tot_obs_time = np.sum(np.diff(T_available)[::2]) * 86400.
    # compute the number of time slots, and their starting times in days.
    # Note that the number of slots per window must be an integer, thus
    # a the end of each window there might be some remainder time unused
    slots_in_window = (np.diff(T_available)[::2] * 86400./T).astype(int)
    n_slots = np.sum(slots_in_window)
    t0_slot = np.empty(n_slots)
    k = 0
    for i in range(n_windows):
        for j in range(slots_in_window[i]):
            t0_slot[k] = T_available[2 * i] + j * T/86400.
            k = k + 1
    # mark all slots as available
    available_slots = np.ones(n_slots,dtype=bool)

    # the strategy will be a healpix map of observation times
    strategy = np.ones(hp.nside2npix(nside))*np.nan

    # assign the available time slots to the skymap pixels in order
    # of descending sky position probability. Each pixel is assigned
    # the available time slot where the detectability is highest
    for p in descending_prob_idx:
        try: # find the time of best detectability
            detp = np.interp(t0_slot[available_slots],t_detmaps,dm[:,p])
            i_best = np.argmax(detp)
            i0_best = np.arange(len(available_slots))[available_slots][i_best] #original index of the best time slot
        except:
            continue

        # do not assign observation if the best detectability is below the requested limit
        if detp[i_best]<min_detectability:
            continue
        else:
            strategy[ridx[p]]=t0_slot[i0_best] # assign the observation time
            available_slots[i0_best]=False # mark the assigned slot as not available anymore
 
    strategy[np.isnan(strategy)] = 0.0
    return strategy
    #return np.ma.masked_invalid(strategy)

def sky_pos_cond_prob(x,x0,map_struct,nside=256,limit_to_region=None):

        npix = hp.nside2npix(nside)
        # precompute mask that selects x_i>x0
        #gt_x0 = x>x0

        # create an empty map
        m = np.ones(npix)*np.nan

        # if no limit_to_region given, create a mask with True everywhere,
        # otherwise adapt the given region resolution to ours
        if limit_to_region is None:
                region = np.ones(npix,dtype=bool)
        else:
                limit_to_region_ones = np.zeros(len(limit_to_region))
                limit_to_region_ones[np.where(limit_to_region==True)[0]] = 1.0
                region = hp.ud_grade(limit_to_region_ones,nside)
                region = np.ceil(region)
                region = region.astype(bool)

        distmu = hp.ud_grade(map_struct["distmu"],nside)
        distnorm = hp.ud_grade(map_struct["distnorm"],nside) 
        distsigma = hp.ud_grade(map_struct["distsigma"],nside)

        # to compute the angular distances of all posterior samples to all pixels of the
        # skymap is quite memory intensive, so we better do it one skymap pixel at a time
        r = np.linspace(1, 2000, 2000)

        for p in np.arange(npix)[region]:
                dp_dr = r**2 * distnorm[p] * norm(distmu[p],distsigma[p]).pdf(r)
                dp_dr_norm = np.cumsum(dp_dr / np.sum(dp_dr))

                app_m = x + 5*(np.log10(r*1e6) - 1) 
                idx = np.argmin(np.abs(app_m-x0))
               
                m[p] = dp_dr_norm[idx]
        
        #return the map, masking invalid values
        m[np.isnan(m)] = 0.0
        return m

# sky_pos_cond_prob_gt, i.e. sky-position-conditional probability that x is greater than x0. 
# In a more compact form: P(x>x0|ra,dec)
def sky_pos_cond_prob_gt(x,x0,ra,dec,nside=256,limit_to_region=None):
	"""
	
	Return a numpy array which represents the healpix projection of the 
	sky-position-conditional probabilities that x>x0, i.e. P(x>x0|sky pos,...),
	constructed from a set of posterior samples as described in Salafia+17
	
	
	Parameters
	----------
	x: numpy 1D array
	    Value of x for each posterior sample.
	x0: float
	    Comparison value.
	ra, dec: numpy arrays of the same length as x
	    Sky positions of the posterior samples.
	nside: int, default=32
	    Resolution of the output healpix map.
	limit_to_region: 1D numpy array of booleans
	    A healpix map of booleans representing the region where the probability has to be computed.
	    Typically, one wants to limit the computation to the 90% confidence area not to waste
	    computational resources. If its nside is different from that given, it will be converted.
	    Default: None (i.e. all-sky).
	
	Returns
	-------
	m: a masked 1D numpy array of length 12*nside**2, which represents the output healpix map.
	"""
	
	# assert that x, ra and dec lengths match
	assert len(x) == len(ra)," The x and ra arrays must have the same lengths"
	assert len(x) == len(dec)," The x and dec arrays must have the same lengths"
	
	# compute the cartesian components of the vectors pointing at the positions
	# of the posterior samples
	npsamp = len(x)
	vx = np.cos(dec)*np.cos(ra)
	vy = np.cos(dec)*np.sin(ra)
	vz = np.sin(dec)
	
	# compute the cartesian components of the vectors pointing at the positions
	# of the skymap pixels
	npix = hp.nside2npix(nside)
	px,py,pz = hp.pix2vec(nside,np.arange(npix))
	
	# precompute mask that selects x_i>x0
	gt_x0 = x>x0

	# create an empty map
	m = np.ones(npix)*np.nan
	
	# if no limit_to_region given, create a mask with True everywhere,
	# otherwise adapt the given region resolution to ours
	if limit_to_region is None:
		region = np.ones(npix,dtype=bool)
	else:
		limit_to_region_ones = np.zeros(len(limit_to_region))
		limit_to_region_ones[np.where(limit_to_region==True)[0]] = 1.0
		
		region = hp.ud_grade(limit_to_region_ones,nside)
		region = np.ceil(region)
		region = region.astype(bool)

	# to compute the angular distances of all posterior samples to all pixels of the
	# skymap is quite memory intensive, so we better do it one skymap pixel at a time
	for p in np.arange(npix)[region]:
		# compute angular distances
		d = np.arccos(px[p]*vx + py[p]*vy + pz[p]*vz)
		
		#compute IDW weigths and normalize them
		bandwidth = np.std(d)*npsamp**(-1./5.) #Silverman's rule (FIXME: use a better estimate?)
		w = np.exp(-0.5*(d/bandwidth)**2)
	
		if np.sum(w)>0.: #do not normalize if the sum is 0
			w = w/np.sum(w)
	
		#compute the probability as prescribed in Salafia+17
		m[p] = np.sum(w[gt_x0])
	
	#return the map, masking invalid values
	m[np.isnan(m)] = 0.0
	return m
	#return np.ma.masked_invalid(m)
