"""
Module to fetch event info and skymap from GraceDB
"""
import os
from pathlib import Path

import lxml.etree
import requests
from ligo.gracedb.rest import GraceDb

from gwemopt.paths import SKYMAP_DIR


def get_event(
    event_name: str,
    output_dir: Path = SKYMAP_DIR,
    rev=None,
):
    """
    Fetches the event info and skymap from GraceDB

    :param event_name: name of the event
    :param output_dir: directory to save the skymap and event info
    :param rev: revision number of the event
    """

    ligo_client = GraceDb()

    voevents = ligo_client.voevents(event_name).json()["voevents"]

    if rev is None:
        rev = len(voevents)

    elif rev > len(voevents):
        raise Exception("Revision {0} not found".format(rev))

    latest_voevent = voevents[rev - 1]
    print(f"Found voevent {latest_voevent['filename']}")

    if "Retraction" in latest_voevent["filename"]:
        raise ValueError(
            f"The specified LIGO event, "
            f"{latest_voevent['filename']}, was retracted."
        )

    response = requests.get(latest_voevent["links"]["file"])

    root = lxml.etree.fromstring(response.content)
    params = {
        elem.attrib["name"]: elem.attrib["value"] for elem in root.iterfind(".//Param")
    }

    latest_skymap = params["skymap_fits"]

    print(f"Latest skymap URL: {latest_skymap}")

    base_file_name = os.path.basename(latest_skymap)
    savepath = output_dir.joinpath(
        f"{event_name}_{latest_voevent['N']}_{base_file_name}",
    )

    if savepath.exists():
        print(f"File {savepath} already exists. Using this.")
    else:
        print(f"Saving to: {savepath}")
        response = requests.get(latest_skymap)

        with open(savepath, "wb") as f:
            f.write(response.content)

    return savepath
