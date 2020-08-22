
import os, sys, optparse
import numpy as np
import healpy as hp
import json

from datetime import datetime
from astropy.time import Time

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.pyplot as plt

import ligo.gracedb
from ligo.gracedb.rest import GraceDb, HTTPError
#from ligo.gracedb.rest import GraceDbBasic

url = 'https://gracedb.ligo.org/api/'
#url = 'https://gracedb-test.ligo.org/apibasic/'

def get_event(params):

    g = GraceDbBasic()
    eventString = params["event"]
    event = g.superevent(params["event"])
    preferred_event = g.event(event.json()["preferred_event"])
    jsonfile = os.path.join(params["outputDir"], 'data.json')
    with open(jsonfile, 'w') as outfile:
        json.dump(preferred_event.json(), outfile)    

    with open(jsonfile, 'r') as f:
        eventinfo = json.load(f)

    event_files = g.files(params["event"]).json()
    for filename in list(event_files):
        if "fits.gz" in filename:
            outfilename = os.path.join(params["outputDir"], filename)
            outfile = open(outfilename,'wb')
            r = g.files(params["event"], filename)
            outfile.write(r.read())
            outfile.close()
            skymapfile = outfilename

    return skymapfile, eventinfo
 
