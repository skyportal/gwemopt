
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
from ligo.gracedb.rest import GraceDbBasic

url = 'https://gracedb-test.ligo.org/api/'
#url = 'https://gracedb-test.ligo.org/apibasic/'

def get_event(params):

    g = GraceDb()
    eventString = params["event"]
    events = g.events('%s'%eventString)
    event = [x for x in events][0]

    keys = ['graceid','gpstime','extra_attributes','group','links','created','far','instruments','labels','nevents','submitter','search','likelihood']
    fileorder = ['LALInference_skymap.fits.gz','bayestar.fits.gz','BW_skymap.fits','LIB_skymap.fits.gz','skyprobcc_cWB.fits']
    #fileorder = ['LALInference3d.fits.gz','bayestar3d.fits.gz','bayestar.fits.gz']

    fileorder = ['LALInference_skymap.fits.gz','bayestar.fits.gz','BW_skymap.fits','LIB_skymap.fits.gz','skyprobcc_cWB.fits']

    eventinfo = {}
    for key in keys:
        if not key in event: continue
        eventinfo[key] = event[key]
    eventinfo['gpstime'] = float(eventinfo['gpstime'])
    if eventinfo['far'] == None:
        eventinfo['far'] = np.nan

    triggerfile = "%s/%s.txt"%(params["outputDir"],eventinfo['graceid'])
    skymapfile = '%s/%s.fits'%(params["outputDir"],eventinfo['graceid'])
    #if os.path.isfile(triggerfile) and os.path.isfile(skymapfile):
    #    print("Already have info for %s... continuing."%event["graceid"])
    #    return
    print("Getting info for %s"%event["graceid"])

    mjds = [-1,-1]
    timediff = -1
    if 'CoincInspiral' in event['extra_attributes']:
        eventinfo['coinc'] = event['extra_attributes']['CoincInspiral']
    if 'SingleInspiral' in event['extra_attributes']:
        eventinfo['singles'] = {}
        for single in event['extra_attributes']['SingleInspiral']:
            eventinfo['singles'][single['ifo']] = single
            eventinfo['singles'][single['ifo']]['gpstime'] = single['end_time']+10**-9 * single['end_time_ns']

        if ("H1" in eventinfo['singles']) and ("L1" in eventinfo['singles']):
            eventinfo["H1_L1_difference"] = eventinfo['singles']['H1']["gpstime"] - eventinfo['singles']['L1']["gpstime"]
            t = Time([eventinfo['singles']['H1']["gpstime"],eventinfo['singles']['L1']["gpstime"]], format='gps', scale='utc')
            mjds = t.mjd
            timediff = eventinfo["H1_L1_difference"]

    if 'MultiBurst' in event['extra_attributes']:
        eventinfo['burst'] = event['extra_attributes']['MultiBurst']

        single_ifo_times = eventinfo['burst']['single_ifo_times'].split(",")           
        ifos = eventinfo['burst']['ifos'].split(",")

        if len(ifos) > 1 and len(single_ifo_times) > 1:
            ifo1 = ifos[0]
            gps1 = float(single_ifo_times[0])
    
            ifo2 = ifos[1]
            gps2 = float(single_ifo_times[1])
    
            eventinfo['burst'][ifo1] = {}
            eventinfo['burst'][ifo1]['gpstime'] = gps1

            eventinfo['burst'][ifo2] = {}
            eventinfo['burst'][ifo2]['gpstime'] = gps2
    
            if ("H1" in eventinfo['burst']) and ("L1" in eventinfo['burst']):
                eventinfo["H1_L1_difference"] = eventinfo['burst']['H1']["gpstime"] - eventinfo['burst']['L1']["gpstime"]
                t = Time([eventinfo['burst']['H1']["gpstime"],eventinfo['burst']['L1']["gpstime"]], format='gps', scale='utc')
                mjds = t.mjd
                timediff = eventinfo["H1_L1_difference"]
    
    try:       
        print("Looking for EM bright file...")
        r = g.files(eventinfo['graceid'], "Source_Classification_%s.json"%eventinfo['graceid'])
        emfile = open('embright.json','w')
        emfile.write(r.read())
        emfile.close()

        with open('embright.json') as data_file:    
            emdata = json.load(data_file)          

        os.system('rm embright.json')

        embright_keys = ["Prob remnant_mass_greater_than 0M_sun","Prob EMbright"]  
        ns_keys = ["Prob Mass2_less_than 3M_sun","Prob NS2"]

        embright_prob = -1
        for key in embright_keys:
            if not key in embright_keys: continue
            embright_prob = emdata[key]
            break
        ns_prob = -1
        for key in embright_keys: 
            if not key in embright_keys: continue
            ns_prob = emdata[key]
            break

        eventinfo['embright'] = {}
        eventinfo['embright']['embright'] = embright_prob
        eventinfo['embright']['ns'] = ns_prob

    except:
         print("No EM bright file...")            

    try:
        print("Looking for cWB file...")
        r = g.files(eventinfo['graceid'], "trigger_%.4f.txt"%eventinfo['gpstime']) 
        # r = g.files(eventinfo['graceid'], "eventDump.txt")
        cwbfile = open('trigger.txt','w')
        cwbfile.write(r.read())
        cwbfile.close()

        eventinfo['burst'] = {}
        lines = [line.rstrip('\n') for line in open('trigger.txt')]
        for line in lines:
            lineSplit = line.split(":")
            if len(lineSplit) < 2: continue
            key = lineSplit[0]
            value = filter(None, lineSplit[1].split(" "))
            eventinfo['burst'][lineSplit[0]] = value

        ifo1 = eventinfo['burst']['ifo'][0]
        gps1 = float(eventinfo['burst']['time'][0])

        ifo2 = eventinfo['burst']['ifo'][1]
        gps2 = float(eventinfo['burst']['time'][1])

        eventinfo['burst'][ifo1] = {}
        eventinfo['burst'][ifo1]['gpstime'] = gps1

        eventinfo['burst'][ifo2] = {}
        eventinfo['burst'][ifo2]['gpstime'] = gps2

        if ("H1" in eventinfo['burst']) and ("L1" in eventinfo['burst']):
            eventinfo["H1_L1_difference"] = eventinfo['burst']['H1']["gpstime"] - eventinfo['burst']['L1']["gpstime"]
            t = Time([eventinfo['burst']['H1']["gpstime"],eventinfo['burst']['L1']["gpstime"]], format='gps', scale='utc')
            mjds = t.mjd
            timediff = eventinfo["H1_L1_difference"]

    except:
        print("No cWB file...")

    ra = 0
    dec = 0

    r = []
    for lvfile in fileorder:
        #r = g.files(eventinfo['graceid'], lvfile)
        try:
            r = g.files(eventinfo['graceid'], lvfile)
            break
        except:
            continue
    if r == []:
        print("Download of skymaps file for %s failed..."%eventinfo['graceid'])
    else:
   
        skymap = open(skymapfile,'w')
        skymap.write(r.read())
        skymap.close()
      
    return skymapfile, eventinfo
 
