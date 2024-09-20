"""
Module to fetch event info and skymap from GraceDB, url, or locally
"""

import os
from pathlib import Path

import astropy_healpix as ah
import healpy as hp
import ligo.skymap.distance as ligodist
import ligo.skymap.plot
import lxml.etree
import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from ligo.gracedb.rest import GraceDb
from ligo.skymap import moc
from ligo.skymap.bayestar import derasterize, rasterize
from ligo.skymap.io import read_sky_map
from matplotlib import pyplot as plt
from mocpy import MOC
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
        order, _ = moc.uniq2nest(skymap[maP_idx]["UNIQ"])
        nside = hp.order2nside(order)
        # get the nested idx for the given sky location
        nest_idx = hp.ang2pix(nside, theta, phi, nest=True)
        # find the row with the closest nested index
        nest_idxs = []
        for row in skymap:
            order_per_row, nest_idx_per_row = moc.uniq2nest(row["UNIQ"])
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
        order, nest_idx = moc.uniq2nest(uniq_idx)
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

        map_struct = {}

        filename = params["skymap"]

        if params["do_3d"]:
            skymap = read_sky_map(filename, moc=True, distances=True)

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

            map_struct["skymap"] = skymap
        else:
            skymap = read_sky_map(filename, moc=True, distances=False)
            map_struct["skymap"] = skymap

    level, ipix = ah.uniq_to_level_ipix(map_struct["skymap"]["UNIQ"])
    LEVEL = MOC.MAX_ORDER
    shift = 2 * (LEVEL - level)
    hpx = np.array(np.vstack([ipix << shift, (ipix + 1) << shift]), dtype=np.uint64).T
    nside = ah.level_to_nside(level)
    pixel_area = ah.nside_to_pixel_area(ah.level_to_nside(level))
    ra, dec = ah.healpix_to_lonlat(ipix, nside, order="nested")
    map_struct["skymap"]["ra"] = ra.deg
    map_struct["skymap"]["dec"] = dec.deg

    map_struct["skymap_raster"] = rasterize(
        map_struct["skymap"], order=hp.nside2order(int(params["nside"]))
    )

    peak = map_struct["skymap_raster"][
        map_struct["skymap_raster"]["PROB"]
        == np.max(map_struct["skymap_raster"]["PROB"])
    ]
    map_struct["center"] = SkyCoord(peak["ra"][0] * u.deg, peak["dec"][0] * u.deg)

    map_struct["skymap_schedule"] = map_struct["skymap"].copy()

    if params["galactic_limit"] > 0.0:
        coords = SkyCoord(ra=ra, dec=dec)
        ipix = np.where(np.abs(coords.galactic.b.deg) <= params["galactic_limit"])[0]
        map_struct["skymap_schedule"]["PROBDENSITY"][ipix] = 0.0

    map_struct["skymap_raster_schedule"] = rasterize(
        map_struct["skymap_schedule"], order=hp.nside2order(int(params["nside"]))
    )

    if params["confidence_level"] < 1.0:
        prob = map_struct["skymap_raster_schedule"]["PROB"]
        ind = np.argsort(prob)[::-1]
        prob = prob[ind]
        cumprob = np.cumsum(prob)
        ii = np.where(cumprob > params["confidence_level"])[0]
        map_struct["skymap_raster_schedule"]["PROB"][ind[ii]] = 0.0
        map_struct["skymap_schedule"] = derasterize(map_struct["skymap_raster"].copy())

    if "DISTMU" in map_struct["skymap_raster"].columns:
        (
            map_struct["skymap_raster"]["DISTMEAN"],
            map_struct["skymap_raster"]["DISTSTD"],
            mom_norm,
        ) = ligodist.parameters_to_moments(
            map_struct["skymap_raster"]["DISTMU"],
            map_struct["skymap_raster"]["DISTSIGMA"],
        )

    extra_header = [
        ("PIXTYPE", "HEALPIX", "HEALPIX pixelisation"),
        ("ORDERING", "NESTED", "Pixel ordering scheme: RING, NESTED, or NUNIQ"),
        ("COORDSYS", "C", "Ecliptic, Galactic or Celestial (equatorial)"),
        (
            "MOCORDER",
            moc.uniq2order(map_struct["skymap"]["UNIQ"].max()),
            "MOC resolution (best order)",
        ),
        ("INDXSCHM", "EXPLICIT", "Indexing: IMPLICIT or EXPLICIT"),
    ]

    hdu = fits.table_to_hdu(map_struct["skymap_raster"])
    hdu.header.extend(extra_header)
    map_struct["hdu"] = hdu

    return params, map_struct
