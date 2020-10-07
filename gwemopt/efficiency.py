
import os, sys
import numpy as np
import healpy as hp
from astropy.time import Time
import scipy.stats
from scipy.interpolate import interpolate as interp

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.pyplot as plt

from ligo.skymap import distance

import gwemopt.utils

def compute_efficiency(params, map_struct, lightcurve_struct, coverage_struct,
                       do3D=False):

    nside = params["nside"]
    npix = hp.nside2npix(nside)

    Ninj = params["Ninj"]
    Ndet = params["Ndet"]
    gpstime = params["gpstime"]
    mjd_inj = Time(gpstime, format='gps', scale='utc').mjd
    #FOV_r = np.sqrt(float(params['FOV'])/np.pi)

    if params["doCatalog"]:
        distn = scipy.stats.rv_discrete(values=(np.arange(npix),
                                                map_struct["prob_catalog"]))
    else:
        distn = scipy.stats.rv_discrete(values=(np.arange(npix),
                                                map_struct["prob"]))
    ipix = distn.rvs(size=Ninj)
    ras, decs = hp.pix2ang(nside, ipix, lonlat=True)

    dists = np.logspace(-1,3,1000)
    ndetections = np.zeros((len(dists),))

    for pinpoint in ipix:
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
            idx = np.where(np.isfinite(lightcurve_mag))[0]

            f = interp.interp1d(lightcurve_t[idx], lightcurve_mag[idx],
                                fill_value='extrapolate')
            lightcurve_mag_interp = f(mjd)
            dist_threshold = (10**(((mag-lightcurve_mag_interp)/5.0)+1.0))/1e6

            idxs_detections = np.where(dists <= dist_threshold)[0]
            detections[idxs_detections] = detections[idxs_detections] + 1

        idxs_detections = np.where(detections >= Ndet)[0]
        ndetections[idxs_detections] = ndetections[idxs_detections] + 1

    efficiency = ndetections / Ninj
    efficiency_struct = {}
    efficiency_struct["ra"] = ras
    efficiency_struct["dec"] = decs
    efficiency_struct["efficiency"] = efficiency
    efficiency_struct["distances"] = dists

    save_efficiency_data(params, efficiency_struct, lightcurve_struct)

    if do3D:
        eff_3D,dists_inj = compute_3d_efficiency(params, map_struct, lightcurve_struct, coverage_struct)
        efficiency_metric = [eff_3D, np.sqrt(eff_3D * (1 - eff_3D) / params["Ninj"])]
        save_efficiency_metric(params, os.path.join(params["outputDir"],"efficiency.txt"), efficiency_metric, lightcurve_struct)
        efficiency_struct["3D"] = eff_3D
        efficiency_struct["dists_inj"] = dists_inj

    if params["doTrueLocation"]:
        det = compute_true_efficiency(params, map_struct, lightcurve_struct, coverage_struct)
        filename = os.path.join(params["outputDir"],'efficiency_true_' + lightcurve_struct['name'] + '.txt')
   
        fid = open(filename, 'w')
        if det:
            fid.write('1')
        else:
            fid.write('0') 
        fid.close()

    return efficiency_struct

def compute_true_efficiency(params, map_struct, lightcurve_struct, coverage_struct):

    nside = params["nside"]
    npix = hp.nside2npix(nside)
    Ninj = params["Ninj"]
    gpstime = params["gpstime"]
    mjd_inj = Time(gpstime, format='gps', scale='utc').mjd

    ipix = hp.ang2pix(nside, params["true_ra"], params["true_dec"], lonlat=True)
    #ipix = np.argmax(map_struct["prob"])
    #lat, lon = hp.pix2ang(nside, ipix, lonlat=True)
    dist = params["true_distance"]

    idxs = []
    for jj in range(len(coverage_struct["ipix"])):
        expPixels = coverage_struct["ipix"][jj]

        if ipix in expPixels:
            idxs.append(jj)

    mjds = coverage_struct["data"][idxs,2]
    mags = coverage_struct["data"][idxs,3]
    filts = coverage_struct["filters"][idxs]
    single_detection = False
    for mjd, mag, filt in zip(mjds,mags,filts):
        lightcurve_t = lightcurve_struct["t"] + mjd_inj
        lightcurve_mag = lightcurve_struct[filt]
        idx = np.where(np.isfinite(lightcurve_mag))[0]

        f = interp.interp1d(lightcurve_t[idx], lightcurve_mag[idx],fill_value='extrapolate')
        lightcurve_mag_interp = f(mjd)
        dist_threshold = (10**(((mag-lightcurve_mag_interp)/5.0)+1.0))/1e6

        if dist<=dist_threshold:
            single_detection = True
            break

    return single_detection


def compute_3d_efficiency(params, map_struct, lightcurve_struct, coverage_struct):

    nside = params["nside"]
    npix = hp.nside2npix(nside)
    Ninj = params["Ninj"]
    gpstime = params["gpstime"]
    mjd_inj = Time(gpstime, format='gps', scale='utc').mjd
    
    if params["doCatalog"]:
        distn = scipy.stats.rv_discrete(values=(np.arange(npix),map_struct["prob_catalog"]))
    else:
        distn = scipy.stats.rv_discrete(values=(np.arange(npix),map_struct["prob"]))
    ipix = distn.rvs(size=Ninj)

    #calculate the moments from distmu, distsigma and distnorm
    mom_mean, mom_std, mom_norm = distance.parameters_to_moments(map_struct["distmu"],map_struct["distsigma"])

    detections = 0
    dists_inj = {}
    dists_inj["recovered"], dists_inj["tot"] = [],[]
    for pinpoint in ipix:
        dist = -1
        while (dist < 0):
            if np.isinf(mom_mean[pinpoint]) or np.isinf(mom_std[pinpoint]):
                dist = np.inf
            else:
                dist = mom_mean[pinpoint] + mom_std[pinpoint] * np.random.normal()

        if dist != np.inf:
            dists_inj["tot"].append(dist)
        
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
        single_detection = False
        for mjd, mag, filt in zip(mjds,mags,filts):
            lightcurve_t = lightcurve_struct["t"] + mjd_inj
            lightcurve_mag = lightcurve_struct[filt]
            idx = np.where(np.isfinite(lightcurve_mag))[0]
            
            f = interp.interp1d(lightcurve_t[idx], lightcurve_mag[idx],fill_value='extrapolate')
            lightcurve_mag_interp = f(mjd)
            dist_threshold = (10**(((mag-lightcurve_mag_interp)/5.0)+1.0))/1e6

            if dist<=dist_threshold:
                single_detection = True
                break

        if single_detection:
            detections+=1
            dists_inj["recovered"].append(dist)

    return detections/Ninj,dists_inj

def save_efficiency_data(params, efficiency_struct, lightcurve_struct):

    filename = os.path.join(params["outputDir"],'efficiency_' + lightcurve_struct['name'] + '.txt')

    for i in range(0, len(efficiency_struct["distances"])):
        dist = efficiency_struct["distances"][i]
        eff = efficiency_struct["efficiency"][i]
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
