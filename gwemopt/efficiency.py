
import os, sys
import numpy as np
import healpy as hp
from astropy.time import Time
import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.pyplot as plt

import gwemopt.utils

def compute_efficiency(params, map_struct, eventinfo, lightcurve_struct, coverage_struct):

    nside = params["nside"]
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

    for ii in range(Ninj):
        # this returns the index of the value in cumsum of probablity closest to a random value
        ipix = np.argmin(np.abs(prob_data_cumsum-rand_values[ii])) 
        # the point corresponding to that
        ra_inj = map_struct["ra"][prob_data_indexes][ipix]
        dec_inj = map_struct["dec"][prob_data_indexes][ipix]

        ras.append(ra_inj)
        decs.append(dec_inj)

        # THE SKY-LOCATION AS A HEALPIXEL ID
        pinpoint = hp.ang2pix(nside, theta=ra_inj, phi=dec_inj, lonlat=True)

        #idxs = np.where(np.sqrt((coverage_struct["data"][:,0]-ra_inj)**2 + (coverage_struct["data"][:,1]-dec_inj)**2) <= FOV_r)[0]
        #idxs_ra = np.where(np.abs(coverage_struct["data"][:,0]-ra_inj)<=coverage_struct["FOV"]/2.0)[0]
        #idxs_dec = np.where(np.abs(coverage_struct["data"][:,1]-dec_inj)<=coverage_struct["FOV"]/2.0)[0]
        #idxs = np.intersect1d(idxs_ra,idxs_dec)

        idxs = []
        for jj in range(len(coverage_struct["ipix"])):
            expPixels = coverage_struct["ipix"][jj]
            if pinpoint in expPixels:
                idxs.append(jj)
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
    save_efficiency_data(params, efficiency_struct, lightcurve_struct)
    efficiency_metric = calculate_efficiency_metric(params, efficiency_struct)
    save_efficiency_metric(params, "efficiency.txt", efficiency_metric, lightcurve_struct)
    return efficiency_struct


def save_efficiency_data(params, efficiency_struct, lightcurve_struct):
    for i in range(0, len(efficiency_struct["distances"])):
        dist = efficiency_struct["distances"][i]
        eff = efficiency_struct["efficiency"][i]
        filename = os.path.join(params["outputDir"],'efficiency_' + lightcurve_struct['name'] + '.txt')
        if os.path.exists(filename):
            append_write = 'a'
            efficiency_file = open(filename, append_write)
        else:
            append_write = 'w'
            efficiency_file = open(filename, append_write)
            efficiency_file.write("Distance" + "\t" + "efficiency\n")
        efficiency_file.write(str(dist) + "\t" + str(eff) + "\n")
    efficiency_file.close()


def calculate_efficiency_metric(params, efficiency_struct):
    dist_sum = 0
    weighted_sum = 0
    for i in range(0, len(efficiency_struct["distances"])):
        dist = efficiency_struct["distances"][i]
        eff = efficiency_struct["efficiency"][i]
        dist_sum += dist * dist
        weighted_sum += dist * dist * eff
    metric = weighted_sum / dist_sum
    uncertainty = np.sqrt(metric * (1 - metric) / params["Ninj"])
    return (metric, uncertainty)


def save_efficiency_metric(params, efficiency_filename, efficiency_metric, lightcurve_struct):
    if os.path.exists(efficiency_filename):
        append_write = 'a'
        efficiency_file = open(efficiency_filename, append_write)
    else:
        append_write = 'w'
        efficiency_file = open(efficiency_filename, append_write)
        efficiency_file.write("tilesType\t" + "timeallocationType\t" + "scheduleType\t" + "Ntiles\t" + "efficiencyMetric\t" + "efficiencyMetric_err\t" + "injection\n")
    efficiency_file.write(params["tilesType"] + "\t" + params["timeallocationType"] + "\t" +  params["scheduleType"] + "\t" + str(params["Ntiles"]) + "\t" + str(efficiency_metric[0]) + "\t" + str(efficiency_metric[1]) + "\t" + lightcurve_struct["name"] + "\n")
    efficiency_file.close()
