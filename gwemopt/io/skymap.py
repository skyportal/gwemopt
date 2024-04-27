"""
Module to fetch event info and skymap from GraceDB, url, or locally
"""

import os
from pathlib import Path

import healpy as hp
import ligo.skymap.distance as ligodist
import lxml.etree
import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from ligo.gracedb.rest import GraceDb
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map
from ligo.skymap.moc import uniq2nest
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

from gwemopt.paths import SKYMAP_DIR


def download_from_url(skymap_url: str, output_dir: Path, skymap_name: str) -> Path:
    """
    Download a skymap from a URL

    :param skymap_url: URL to download from
    :param output_dir: Output directory
    :param skymap_name: Name of skymap
    :return: Path to downloaded skymap
    """
    savepath = output_dir.joinpath(skymap_name)

    if savepath.exists():
        print(f"File {savepath} already exists. Using this.")
    else:
        print(f"Saving to: {savepath}")
        response = requests.get(skymap_url, headers={"User-Agent": "Mozilla/5.0"})

        with open(savepath, "wb") as f:
            f.write(response.content)

    return savepath


def get_skymap_gracedb(
    event_name: str, rev=None, output_dir: Path = SKYMAP_DIR
) -> Path:
    """
    Fetches the skymap from GraceDB

    :param event_name: name of the event
    :param rev: revision number of the event
    :param output_dir: directory to save the skymap and event info
    :return: path to the skymap
    """
    ligo_client = GraceDb()

    voevents = ligo_client.voevents(event_name).json()["voevents"]

    if rev is None:
        rev = len(voevents)

    elif rev > len(voevents):
        raise Exception(f"Revision {0} not found".format(rev))

    latest_voevent = voevents[rev - 1]
    print(f"Found voevent {latest_voevent['filename']}")

    if "Retraction" in latest_voevent["filename"]:
        raise ValueError(
            f"The specified LIGO event, "
            f"{latest_voevent['filename']}, was retracted."
        )

    response = requests.get(
        latest_voevent["links"]["file"],
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )

    root = lxml.etree.fromstring(response.content)
    params = {
        elem.attrib["name"]: elem.attrib["value"] for elem in root.iterfind(".//Param")
    }

    latest_skymap_url = params["skymap_fits"]

    print(f"Latest skymap URL: {latest_skymap_url}")

    skymap_name = "_".join(
        [event_name, str(latest_voevent["N"]), os.path.basename(latest_skymap_url)]
    )

    skymap_path = download_from_url(latest_skymap_url, output_dir, skymap_name)

    return skymap_path


def get_skymap(event_name: str, output_dir: Path = SKYMAP_DIR, rev: int = None) -> Path:
    """
    Fetches the event info and skymap from GraceDB

    :param event_name: name of the event
    :param output_dir: directory to save the skymap and event info
    :param rev: revision number of the event
    :return: path to the skymap
    """

    if Path(event_name).exists():
        savepath = Path(event_name)
    elif output_dir.joinpath(event_name).exists():
        savepath = output_dir.joinpath(event_name)
    elif event_name[:8] == "https://":
        savepath = download_from_url(
            event_name, output_dir, os.path.basename(event_name)
        )
    else:
        savepath = get_skymap_gracedb(event_name, output_dir=output_dir, rev=rev)

    return savepath


def read_inclination(skymap, params, map_struct):
    # check if the sky location is input
    # if not, the maximum posterior point is taken
    if params["true_location"]:
        ra = params["true_ra"]
        dec = params["true_dec"]
        print(f"Using the input sky location ra={ra}, dec={dec}")
        # convert them back to theta and phi
        phi = np.deg2rad(ra)
        theta = 0.5 * np.pi - np.deg2rad(dec)
        # make use of the maP nside
        maP_idx = np.argmax(skymap["PROBDENSITY"])
        order, _ = uniq2nest(skymap[maP_idx]["UNIQ"])
        nside = hp.order2nside(order)
        # get the nested idx for the given sky location
        nest_idx = hp.ang2pix(nside, theta, phi, nest=True)
        # find the row with the closest nested index
        nest_idxs = []
        for row in skymap:
            order_per_row, nest_idx_per_row = uniq2nest(row["UNIQ"])
            if order_per_row == order:
                nest_idxs.append(nest_idx_per_row)
            else:
                nest_idxs.append(0)
        nest_idxs = np.array(nest_idxs)
        row = skymap[np.argmin(np.absolute(nest_idxs - nest_idx))]
    else:
        print("Using the maP point from the fits file input")
        maP_idx = np.argmax(skymap["PROBDENSITY"])
        uniq_idx = skymap[maP_idx]["UNIQ"]
        # convert to nested indexing and find the location of that index
        order, nest_idx = uniq2nest(uniq_idx)
        nside = hp.order2nside(order)
        theta, phi = hp.pix2ang(nside, int(nest_idx), nest=True)
        # convert theta and phi to ra and dec
        ra = np.rad2deg(phi)
        dec = np.rad2deg(0.5 * np.pi - theta)
        print(f"The maP location is ra={ra}, dec={dec}")
        # fetching the skymap row
        row = skymap[maP_idx]

    # construct the iota prior
    cosiota_nodes_num = 10
    cosiota_nodes = np.cos(np.linspace(0, np.pi, cosiota_nodes_num))
    colnames = ["PROBDENSITY", "DISTMU", "DISTSIGMA", "DISTNORM"]
    # do an all-in-one interpolation
    prob_density, dist_mu, dist_sigma, dist_norm = (
        PchipInterpolator(
            cosiota_nodes[::-1],
            row["{}_SAMPLES".format(colname)][::-1],
        )
        for colname in colnames
    )
    # now have the joint distribution evaluated
    u = np.linspace(-1, 1, 1000)  # this is cosiota
    # fetch the fixed distance
    dL = params["true_distance"]
    prob_u = (
        prob_density(u)
        * dist_norm(u)
        * np.square(dL)
        * norm(dist_mu(u), dist_sigma(u)).pdf(dL)
    )

    iota = np.arccos(u)
    prob_iota = prob_u * np.absolute(np.sin(iota))
    # in GW, iota in [0, pi], but in EM, iota in [0, pi/2]
    # therefore, we need to do a folding
    # split the domain in half
    iota_lt_pi2 = iota[iota < np.pi / 2]
    prob_lt_pi2, prob_gt_pi2 = prob_iota[iota < np.pi / 2], prob_iota[iota >= np.pi / 2]
    iota_EM = iota_lt_pi2
    prob_iota_EM = prob_lt_pi2 + prob_gt_pi2[::-1]

    # normalize
    prob_iota /= np.trapz(iota_EM, prob_iota_EM)

    map_struct["iota_EM"] = iota_EM
    map_struct["prob_iota_EM"] = prob_iota_EM

    map_struct["prob_density_interp"] = prob_density
    map_struct["dist_mu_interp"] = dist_mu
    map_struct["dist_sigma_interp"] = dist_sigma
    map_struct["dist_norm_interp"] = dist_norm

    return map_struct


def read_skymap(params, map_struct=None):
    """
    Read in a skymap and return a map_struct

    :param params: dictionary of parameters
    :param map_struct: dictionary of map parameters
    :return: map_struct
    """

    geometry = params["geometry"]
    if geometry is not None:
        if geometry == "2d":
            params["do_3d"] = False
        else:
            params["do_3d"] = True

    if map_struct is None:
        # Let's just figure out what's in the skymap first
        skymap_path = params["skymap"]

        params["name"] = Path(skymap_path).stem

        is_3d = False
        t_obs = Time.now()

        with fits.open(skymap_path) as hdul:
            for x in hdul:
                if "DATE-OBS" in x.header:
                    t_obs = Time(x.header["DATE-OBS"], format="isot")

                elif "EVENTMJD" in x.header:
                    t_obs_mjd = x.header["EVENTMJD"]
                    t_obs = Time(t_obs_mjd, format="mjd")

                if ("DISTMEAN" in x.header) | ("DISTSTD" in x.header):
                    is_3d = True

        # Set GPS time from skymap, if not specified. Defaults to today
        params["eventtime"] = t_obs
        if params["gpstime"] is None:
            params["gpstime"] = t_obs.gps

        if "do_3d" not in params:
            params["do_3d"] = is_3d

        header = []
        if map_struct is None:
            map_struct = {}

            filename = params["skymap"]

            if params["do_3d"]:
                try:
                    healpix_data, header = hp.read_map(
                        filename, field=(0, 1, 2, 3), h=True
                    )
                except:
                    skymap = read_sky_map(filename, moc=True, distances=True)
                    order = hp.nside2order(params["nside"])

                    # for colname in skymap.colnames:
                    #    if colname.startswith('PROB'):
                    #        newname = colname.replace('PROB', 'PROBDENSITY')
                    #        skymap.rename_column(colname, newname)
                    #        skymap[newname] *= len(skymap) / (4 * np.pi)
                    #        skymap[newname].unit = u.steradian ** -1

                    if "PROBDENSITY_SAMPLES" in skymap.columns:
                        if params["inclination"]:
                            map_struct = read_inclination(skymap, params, map_struct)

                        skymap.remove_columns(
                            [
                                f"{name}_SAMPLES"
                                for name in [
                                    "PROBDENSITY",
                                    "DISTMU",
                                    "DISTSIGMA",
                                    "DISTNORM",
                                ]
                            ]
                        )

                    t = rasterize(skymap)
                    result = t["PROB"], t["DISTMU"], t["DISTSIGMA"], t["DISTNORM"]
                    healpix_data = hp.reorder(result, "NESTED", "RING")

                distmu_data = healpix_data[1]
                distsigma_data = healpix_data[2]
                prob_data = healpix_data[0]
                norm_data = healpix_data[3]

                map_struct["distmu"] = distmu_data / params["DScale"]
                map_struct["distsigma"] = distsigma_data / params["DScale"]
                map_struct["prob"] = prob_data
                map_struct["distnorm"] = norm_data

            else:
                prob_data, header = hp.read_map(
                    filename, field=0, verbose=False, h=True
                )
                prob_data = prob_data / np.sum(prob_data)

                map_struct["prob"] = prob_data

        for j in range(len(header)):
            if header[j][0] == "DATE":
                map_struct["trigtime"] = header[j][1]

    natural_nside = hp.pixelfunc.get_nside(map_struct["prob"])
    nside = params["nside"]

    print("natural_nside =", natural_nside)
    print("nside =", nside)

    if not params["do_3d"]:
        map_struct["prob"] = hp.ud_grade(map_struct["prob"], nside, power=-2)

    if params["do_3d"]:
        if natural_nside != nside:
            map_struct["prob"] = hp.pixelfunc.ud_grade(
                map_struct["prob"], nside, power=-2
            )
            map_struct["distmu"] = hp.pixelfunc.ud_grade(map_struct["distmu"], nside)
            map_struct["distsigma"] = hp.pixelfunc.ud_grade(
                map_struct["distsigma"], nside
            )
            map_struct["distnorm"] = hp.pixelfunc.ud_grade(
                map_struct["distnorm"], nside
            )

            map_struct["distmu"][map_struct["distmu"] < -1e30] = np.inf

        nside_down = 32

        distmu_down = hp.pixelfunc.ud_grade(map_struct["distmu"], nside_down)

        (
            map_struct["distmed"],
            map_struct["diststd"],
            mom_norm,
        ) = ligodist.parameters_to_moments(
            map_struct["distmu"], map_struct["distsigma"]
        )

        distmu_down[distmu_down < -1e30] = np.inf

        map_struct["distmed"] = hp.ud_grade(map_struct["distmed"], nside, power=-2)
        map_struct["diststd"] = hp.ud_grade(map_struct["diststd"], nside, power=-2)

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5 * np.pi - theta)

    map_struct["ra"] = ra
    map_struct["dec"] = dec

    if params["doRASlice"]:
        ra_low, ra_high = params["raslice"][0], params["raslice"][1]
        if ra_low <= ra_high:
            ipix = np.where(
                (ra_high * 360.0 / 24.0 < ra) | (ra_low * 360.0 / 24.0 > ra)
            )[0]
        else:
            ipix = np.where(
                (ra_high * 360.0 / 24.0 < ra) & (ra_low * 360.0 / 24.0 > ra)
            )[0]
        map_struct["prob"][ipix] = 0.0
        map_struct["prob"] = map_struct["prob"] / np.sum(map_struct["prob"])

    if params["galactic_limit"] > 0.0:
        coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        ipix = np.where(np.abs(coords.galactic.b.deg) <= params["galactic_limit"])[0]
        map_struct["prob"][ipix] = 0.0
        map_struct["prob"] = map_struct["prob"] / np.sum(map_struct["prob"])

    sort_idx = np.argsort(map_struct["prob"])[::-1]
    csm = np.empty(len(map_struct["prob"]))
    csm[sort_idx] = np.cumsum(map_struct["prob"][sort_idx])

    map_struct["cumprob"] = csm
    map_struct["ipix_keep"] = np.where(csm <= params["iterativeOverlap"])[0]

    pixarea = hp.nside2pixarea(nside)
    pixarea_deg2 = hp.nside2pixarea(nside, degrees=True)

    map_struct["nside"] = nside
    map_struct["npix"] = npix
    map_struct["pixarea"] = pixarea
    map_struct["pixarea_deg2"] = pixarea_deg2

    return params, map_struct
