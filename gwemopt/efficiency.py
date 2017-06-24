
import os, sys

import numpy as np
from astropy.time import Time
import healpy as hp

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.pyplot as plt

def compute_efficiency(params, map_struct, eventinfo, lightcurve_struct, coverage_struct):

    Ninj = params["Ninj"]
    Ndet = params["Ndet"]
    gpstime = eventinfo["gpstime"]
    mjd_inj = Time(gpstime, format='gps', scale='utc').mjd
    #FOV_r = np.sqrt(float(params['FOV'])/np.pi)

    prob_data_sorted = np.sort(map_struct["prob"])[::-1]
    prob_data_indexes = np.argsort(map_struct["prob"])[::-1]
    prob_data_cumsum = np.cumsum(prob_data_sorted)

    rand_values = np.random.rand(Ninj,)
    dists = np.logspace(-1,3,1000)
    ndetections = np.zeros((len(dists),))

    ras = []
    decs = []

    for ii in xrange(Ninj):
        ipix = np.argmin(np.abs(prob_data_cumsum-rand_values[ii]))
        ra_inj = map_struct["ra"][prob_data_indexes][ipix]
        dec_inj = map_struct["dec"][prob_data_indexes][ipix]

        ras.append(ra_inj)
        decs.append(dec_inj)

        #idxs = np.where(np.sqrt((coverage_struct["data"][:,0]-ra_inj)**2 + (coverage_struct["data"][:,1]-dec_inj)**2) <= FOV_r)[0]
        idxs_ra = np.where(np.abs(coverage_struct["data"][:,0]-ra_inj)<=coverage_struct["FOV"]/2.0)[0]
        idxs_dec = np.where(np.abs(coverage_struct["data"][:,1]-dec_inj)<=coverage_struct["FOV"]/2.0)[0]
        idxs = np.intersect1d(idxs_ra,idxs_dec)
        if len(idxs) == 0:
            continue

        mjds = coverage_struct["data"][idxs,2]
        mags = coverage_struct["data"][idxs,3]
        filts = coverage_struct["filters"][idxs]

        detections = np.zeros((len(dists),))

        for mjd, mag, filt in zip(mjds,mags,filts):
            lightcurve_t = lightcurve_struct["t"] + mjd_inj
            lightcurve_mag = lightcurve_struct[filt]

            lightcurve_mag_interp = np.interp(mjd,lightcurve_t,lightcurve_mag)
            dist_threshold = (10**(((mag-lightcurve_mag_interp)/5.0)+1.0))/1e6

            idxs_detections = np.where(dists <= dist_threshold)[0]
            detections[idxs_detections] = detections[idxs_detections] + 1
        idxs_detections = np.where(detections >= Ndet)[0]
        ndetections[idxs_detections] = ndetections[idxs_detections] + 1
    efficiency = ndetections / Ninj
 
    efficiency_struct = {}
    efficiency_struct["ra"] = np.array(ras)
    efficiency_struct["dec"] = np.array(decs)
    efficiency_struct["efficiency"] = efficiency
    efficiency_struct["distances"] = dists

    return efficiency_struct

