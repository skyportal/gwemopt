import os

import astropy.units as u
import healpy as hp
import numpy as np
import scipy.stats
from astropy.coordinates import SkyCoord
from astropy.time import Time
from ligo.skymap import distance
from scipy.interpolate import interp1d

from gwemopt.io.export_efficiency import (
    export_efficiency_data,
    save_detection_nondetection,
    save_efficiency_metric,
)


def compute_efficiency(params, map_struct, lightcurve_struct, coverage_struct):
    nside = params["nside"]
    npix = hp.nside2npix(nside)

    Ninj = params["Ninj"]
    Ndet = params["Ndet"]
    gpstime = params["gpstime"]
    mjd_inj = Time(gpstime, format="gps", scale="utc").mjd

    distn = scipy.stats.rv_discrete(
        values=(np.arange(npix), map_struct["skymap_raster"]["PROB"])
    )
    ipix = distn.rvs(size=Ninj)
    ras, decs = hp.pix2ang(nside, ipix, lonlat=True)

    dists = np.logspace(-1, 3, 1000)
    ndetections = np.zeros((len(dists),))

    for ra, dec in zip(ras, decs):
        coords = SkyCoord(ra * u.deg, dec * u.deg)
        idxs = []
        for jj in range(len(coverage_struct["moc"])):
            moc = coverage_struct["moc"][jj]

            if moc.contains_skycoords(coords):
                idxs.append(jj)
        if len(idxs) == 0:
            continue

        mjds = coverage_struct["data"][idxs, 2]
        mags = coverage_struct["data"][idxs, 3]
        filts = coverage_struct["filters"][idxs]

        detections = np.zeros((len(dists),))

        for mjd, mag, filt in zip(mjds, mags, filts):
            lightcurve_t = lightcurve_struct["t"] + mjd_inj
            lightcurve_mag = lightcurve_struct[filt]
            idx = np.where(np.isfinite(lightcurve_mag))[0]

            f = interp1d(
                lightcurve_t[idx], lightcurve_mag[idx], fill_value="extrapolate"
            )
            lightcurve_mag_interp = f(mjd)
            dist_threshold = (10 ** (((mag - lightcurve_mag_interp) / 5.0) + 1.0)) / 1e6

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

    export_efficiency_data(params, efficiency_struct, lightcurve_struct)

    if params["do_3d"]:
        eff_3D, dists_inj = compute_3d_efficiency(
            params, map_struct, lightcurve_struct, coverage_struct
        )
        efficiency_metric = [eff_3D, np.sqrt(eff_3D * (1 - eff_3D) / params["Ninj"])]
        save_efficiency_metric(
            params,
            os.path.join(params["outputDir"], "efficiency.txt"),
            efficiency_metric,
            lightcurve_struct,
        )
        efficiency_struct["3D"] = eff_3D
        efficiency_struct["dists_inj"] = dists_inj

    if params["true_location"]:
        det = compute_true_efficiency(
            params, map_struct, lightcurve_struct, coverage_struct
        )

        lc_name = lightcurve_struct["name"]
        save_detection_nondetection(
            os.path.join(params["outputDir"], f"efficiency_true_{lc_name}.txt"), det
        )

        efficiency_struct["detection"] = det

    return efficiency_struct


def compute_true_efficiency(params, map_struct, lightcurve_struct, coverage_struct):
    nside = params["nside"]
    npix = hp.nside2npix(nside)
    Ninj = params["Ninj"]
    gpstime = params["gpstime"]
    mjd_inj = Time(gpstime, format="gps", scale="utc").mjd

    dist = params["true_distance"]
    coords = SkyCoord(params["true_ra"] * u.deg, params["true_dec"] * u.deg)
    idxs = []
    for jj in range(len(coverage_struct["moc"])):
        moc = coverage_struct["moc"][jj]

        if moc.contains_skycoords(coords):
            idxs.append(jj)

    mjds = coverage_struct["data"][idxs, 2]
    mags = coverage_struct["data"][idxs, 3]
    filts = coverage_struct["filters"][idxs]
    single_detection = False
    for mjd, mag, filt in zip(mjds, mags, filts):
        lightcurve_t = lightcurve_struct["t"] + mjd_inj
        lightcurve_mag = lightcurve_struct[filt]
        idx = np.where(np.isfinite(lightcurve_mag))[0]

        f = interp.interp1d(
            lightcurve_t[idx], lightcurve_mag[idx], fill_value="extrapolate"
        )
        lightcurve_mag_interp = f(mjd)
        dist_threshold = (10 ** (((mag - lightcurve_mag_interp) / 5.0) + 1.0)) / 1e6

        if dist <= dist_threshold:
            single_detection = True
            break

    return single_detection


def compute_3d_efficiency(params, map_struct, lightcurve_struct, coverage_struct):
    nside = params["nside"]
    npix = hp.nside2npix(nside)
    Ninj = params["Ninj"]
    gpstime = params["gpstime"]
    mjd_inj = Time(gpstime, format="gps", scale="utc").mjd

    distn = scipy.stats.rv_discrete(
        values=(np.arange(npix), map_struct["skymap_raster"]["PROB"])
    )
    ipix = distn.rvs(size=Ninj)
    ras, decs = hp.pix2ang(nside, ipix, lonlat=True)

    detections = 0
    dists_inj = {}
    dists_inj["recovered"], dists_inj["tot"] = [], []
    for pinpoint, ra, dec in zip(ipix, ras, decs):
        coords = SkyCoord(ra * u.deg, dec * u.deg)

        dist = -1
        while dist < 0:
            if np.isinf(map_struct["skymap_raster"]["DISTMEAN"][pinpoint]) or np.isinf(
                map_struct["skymap_raster"]["DISTSTD"][pinpoint]
            ):
                dist = np.inf
            else:
                dist = (
                    map_struct["skymap_raster"]["DISTMEAN"][pinpoint]
                    + map_struct["skymap_raster"]["DISTSTD"][pinpoint]
                    * np.random.normal()
                )

        if dist != np.inf:
            dists_inj["tot"].append(dist)

        idxs = []
        for jj in range(len(coverage_struct["moc"])):
            moc = coverage_struct["moc"][jj]

            if moc.contains_skycoords(coords):
                idxs.append(jj)

        if len(idxs) == 0:
            continue

        mjds = coverage_struct["data"][idxs, 2]
        mags = coverage_struct["data"][idxs, 3]
        filts = coverage_struct["filters"][idxs]
        single_detection = False
        for mjd, mag, filt in zip(mjds, mags, filts):
            lightcurve_t = lightcurve_struct["t"] + mjd_inj
            lightcurve_mag = lightcurve_struct[filt]
            idx = np.where(np.isfinite(lightcurve_mag))[0]

            f = interp1d(
                lightcurve_t[idx], lightcurve_mag[idx], fill_value="extrapolate"
            )
            lightcurve_mag_interp = f(mjd)
            dist_threshold = (10 ** (((mag - lightcurve_mag_interp) / 5.0) + 1.0)) / 1e6

            if dist <= dist_threshold:
                single_detection = True
                break

        if single_detection:
            detections += 1
            dists_inj["recovered"].append(dist)

    return detections / Ninj, dists_inj


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
