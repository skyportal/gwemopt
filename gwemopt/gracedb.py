"""
Module to fetch event info and skymap from GraceDB
"""
import os
import json

from pathlib import Path

from ligo.gracedb.rest import GraceDb
import requests
import lxml.etree

from gwemopt.paths import SKYMAP_DIR


def get_event(
        event_name: str,
        output_dir: Path = SKYMAP_DIR,
        rev: int | None = None,
):
    """
    Fetches the event info and skymap from GraceDB

    :param event_name: name of the event
    :param output_dir: directory to save the skymap and event info
    :param rev: revision number of the event
    """

    # g = GraceDb()
    #
    # event = g.superevent(event_name)
    # preferred_event = g.event(event.json()["preferred_event"])
    # jsonfile = output_dir.joinpath('data.json')
    # with open(jsonfile, 'w') as outfile:
    #     json.dump(preferred_event.json(), outfile)
    #
    # with open(jsonfile, 'r') as f:
    #     eventinfo = json.load(f)
    #
    # event_files = g.files(event_name).json()
    # for filename in list(event_files):
    #     assert "fits.gz" in filename
    #     outfilename = os.path.join(params["outputDir"], filename)
    #     with open(outfilename,'wb') as outfile:
    #         r = g.files(params["event"], filename)
    #         outfile.write(r.read())
    #     skymapfile = outfilename
    #
    # return skymapfile, eventinfo

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
        elem.attrib["name"]: elem.attrib["value"]
        for elem in root.iterfind(".//Param")
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

    return savepath, event_name

 
