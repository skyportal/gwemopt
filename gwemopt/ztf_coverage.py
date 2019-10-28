
import os
import requests
import urllib.parse
from astropy import time
import astropy.units as u
from astropy.table import Table
import numpy as np
import healpy as hp
import pandas as pd

import gwemopt.ztf_tiling

def ztf_queue():

    ZTF_URL = "http://127.0.0.1:9999"
    fields, quadrants = [], []    

    r = requests.get(
        urllib.parse.urljoin(ZTF_URL, 'current_queue'))
    data = r.json()
    queue = pd.read_json(data['queue'], orient='records')
    if len(queue) > 0:
        n_fields = len(queue['field_id'].unique())
        for field in queue['field_id'].unique():
            rcids = np.arange(64)
            for rcid in rcids:
                fields.append(int(field))
                quadrants.append(int(rcid))

    return np.array(fields), np.array(quadrants)


def ztf_depot(start_time=None, end_time=None):
    """ZTF depot reader.

    Ingests information about images from all program ids
    (including program_id = 1) based on the nightly summary.
    This supplements what is available from the TAP interface,
    where information about public images is not available.

    Parameters
    ----------
    start_time : astropy.Time
        Start time of request.
    end_time : astropy.Time)
        End time of request.

    """

    if start_time is None:
        start_time = time.Time.now() - time.TimeDelta(1.0*u.day)
    if end_time is None:
        end_time = time.Time.now()

    depotdir = 'https://ztfweb.ipac.caltech.edu/ztf/depot'

    fields, quadrants = [], []

    mjds = np.arange(np.floor(start_time.mjd), np.ceil(end_time.mjd))
    for mjd in mjds:
        this_time = time.Time(mjd, format='mjd')
        dstr = this_time.iso.split(" ")[0].replace("-", "")

        url = os.path.join(depotdir, '%s/goodsubs_%s.txt' % (dstr, dstr))
        deptable = get_ztf_depot_table(url)
        if len(deptable) == 0:
            continue
       
        obs_grouped_by_jd = deptable.group_by('jd').groups
        for jd, rows in zip(obs_grouped_by_jd.keys, obs_grouped_by_jd):
            obstime = time.Time(rows['jd'][0], format='jd').datetime
            for row in rows:
                if not row["programid"] == 1: continue
                fields.append(int(row['field']))
                quadrants.append(int(row['rcid']))

    return np.array(fields), np.array(quadrants)


def get_ztf_depot_table(url):
    with requests.get(url) as r:
        deptable = Table.read(r.text, format='ascii.fixed_width',
                              data_start=2, data_end=-1)
    return deptable


def get_skymap(params):
   
    nside = params["nside"]

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5*np.pi - theta)

    #fields, quadrants = ztf_queue()
    fields, quadrants = ztf_depot(start_time=params["start_time"],
                                  end_time=params["end_time"])

    tess = params["config"]["ZTF"]["tesselation"]
    ipixs = []
    for field in np.unique(fields):
        idx = np.where(tess[:,0] == field)[0]
        if len(idx) == 0:
            continue
        row = tess[idx[0],:]
        ra, dec = row[1], row[2]
        idx = np.where(field == fields)[0]
        quads = quadrants[idx]
        ipix = gwemopt.ztf_tiling.get_quadrant_ipix(nside, ra, dec,
                                                    subfield_ids=quads)
        if len(ipix) == 0: continue
        ipixs.append(list({y for x in ipix for y in x}))
    ipix = np.array(list({y for x in ipixs for y in x}))

    prob = np.zeros((npix,))
    prob[ipix] = 1.
    prob = prob / np.sum(prob)

    fitsfile = os.path.join(params["outputDir"],'skymap.fits')
    hp.fitsfunc.write_map(fitsfile,prob,overwrite=True)

    return fitsfile
