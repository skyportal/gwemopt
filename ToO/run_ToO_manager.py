"""
Function to run the TOO manager with vo-events files as entry
Author S. Antier - 2022
"""


#!/bin/env/python
# script to test loading of VOEvent, treat them and send them to broker

# import and prepare setup

import sys
#import voeventparse as vp
import lxml
import os

sys.path.insert(0,'./src')
import grandma_manager as grandma_manager


def test_GRB(filename=None):
    """
    Decode CBC like VO event received through GCN
    :param filename: input file name for the VO event in xml format
    :return:
    """

    currentpwd = os.getcwd()
    print('getcwd ' + currentpwd)

    # dir_path linked to json file
    dir_path = os.path.dirname(os.path.abspath(__file__))

    if not filename:
        filename = "./examples/GW_CBC_example.xml"

    # load the file
    with open(folder+element, 'rb') as fid:
        payload = fid.read()
        root = lxml.etree.fromstring(payload)

        file_server = "./inputs/servers_config.json"

        grandma_manager.online_processing(root, file_server, "observation")

    return





#Swift examples
folder = "./examples/swift/"
files=["event_1540061972.4229486_97.xml","event_1540061973.5148437_61.xml","event_1540062039.5789688_67.xml"]

#Fermi examples
#folder = "./examples/fermi/"
#files = ["event_1540061961.5382793_110.xml","event_1540061991.4968317_111.xml", "event_1540062040.1148593_112.xml","event_1540062511.1510792_115.xml"]#,"event_1540034131.3183208_119.xml"]


#LVK examples
#folder = "./exemples/lvk/"
#files = ["GW_Burst_example.xml","GW_CBC_example.xml","GW_Retraction_example.xml"]

for element in files:

    test_GRB(folder+element)

