"""
Module for interacting with the Treasure Map API.
"""
import re
import urllib.parse

import numpy as np
import requests

from gwemopt.utils.pixels import getCirclePixels, getSquarePixels


def get_treasuremap_pointings(params):
    """
    Function to get the pointings from Treasure Map.

    :param params: dictionary of parameters

    """
    BASE = "http://treasuremap.space/api/v0"
    TARGET = "pointings"
    info = {
        "api_token": params["treasuremap_token"],
        "bands": params["filters"],
        "statuses": params["treasuremap_status"],
        "graceid": params["graceid"],
    }

    url = "{}/{}?{}".format(BASE, TARGET, urllib.parse.urlencode(info))

    try:
        r = requests.get(url=url)
    except requests.exceptions.RequestException as e:
        print(e)
        exit(1)

    observations = r.text.split("}")

    # dicts of instrument FOVs
    FOV_square = {44: 4.96, 47: 6.86}
    FOV_circle = {38: 1.1}

    # create coverage_struct
    coverage_struct = {}
    coverage_struct["data"] = np.empty((0, 8))
    coverage_struct["filters"] = []
    coverage_struct["ipix"] = []

    # read information into coverage struct
    for obs in observations:
        if "invalid api_token" in obs:
            print("Invalid Treasure Map API token.")
            exit(1)
        elif "POINT" not in obs:
            continue

        pointing = re.search("\(([^)]+)", obs).group(1)
        pointing = pointing.split(" ")
        ra, dec = float(pointing[0]), float(pointing[1])

        filteridx = obs.find("band") + 10  # jump to starting index of filter
        filter = obs[filteridx:].split('"')[0][:-1]

        instrumentidx = (
            obs.find("instrumentid") + 16
        )  # jump to starting index of instrument id
        instrument_id = int(obs[instrumentidx:].split(",")[0])

        if instrument_id in FOV_square:
            ipix, radecs, patch, area = getSquarePixels(
                ra, dec, FOV_square[instrument_id], params["nside"]
            )
        elif instrument_id in FOV_circle:
            ipix, radecs, patch, area = getCirclePixels(
                ra, dec, FOV_circle[instrument_id], params["nside"]
            )
        else:
            continue

        coverage_struct["data"] = np.append(
            coverage_struct["data"],
            np.array([[ra, dec, -1, -1, -1, -1, -1, -1]]),
            axis=0,
        )
        coverage_struct["filters"].append(filter)
        coverage_struct["ipix"].append(ipix)

    return coverage_struct
