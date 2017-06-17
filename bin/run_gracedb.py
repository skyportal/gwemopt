
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

from scipy.stats import norm

from ligo.gracedb.rest import GraceDb, HTTPError
from ligo.gracedb.rest import GraceDbBasic

url = 'https://gracedb-test.ligo.org/api/'
#url = 'https://gracedb-test.ligo.org/apibasic/'

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("--doGPSLoop",  action="store_true", default=False)
    parser.add_option("--doGPS",  action="store_true", default=False)
    parser.add_option("--doFAR",  action="store_true", default=False)
    parser.add_option("--doType",  action="store_true", default=False)
    parser.add_option("--doLabel",  action="store_true", default=False)
    parser.add_option("--doEvent",  action="store_true", default=False)

    parser.add_option("-f", "--far", help="far threshold",type=float,default=1e-7)
    parser.add_option("-t", "--type", help="trigger type",default="LowMass")
    parser.add_option("-l", "--label", help="trigger label",default="EM_READY & ADVOK & ~INJ")
    parser.add_option("-s", "--startGPS", help="start gps",type=float,default=1135136300)
    parser.add_option("-e", "--endGPS", help="end gps",type=float,default=1135136400)    
    parser.add_option("-o", "--outputDir", help="output directory",default="output")
    #parser.add_option("-r", "--ra", help="ra (deg)",type=float,default=138.30788)
    #parser.add_option("-d", "--dec", help="dec (deg)",type=float,default=61.09263)
    #parser.add_option("-z", "--dist", help="distance (mpc)",type=float,default=990.0)

    parser.add_option("-r", "--ra", help="ra (deg)",type=float,default=122.9671)
    parser.add_option("-d", "--dec", help="dec (deg)",type=float,default=25.4226)
    parser.add_option("-z", "--dist", help="distance (mpc)",type=float,default=982.2)

    parser.add_option("-n", "--event", help="event name",default="G268556")


    opts, args = parser.parse_args()

    return opts

def get_eventstring(opts):
    eventString = ''
    if opts.doEvent:
        eventString = opts.event
        return eventString

    if opts.doGPS:
        eventString = '%s %.1f .. %.1f'%(eventString,opts.startGPS,opts.endGPS)
    if opts.doLabel:
        eventString = '%s %s'%(eventString,opts.label)
    if opts.doType:
        eventString = '%s %s'%(eventString,opts.type)
    if opts.doFAR:
        eventString = '%s far <%.5e'%(eventString,opts.far)
    return eventString

def download_events(events):

    keys = ['graceid','gpstime','extra_attributes','group','links','created','far','instruments','labels','nevents','submitter','search','likelihood']
    fileorder = ['LALInference_skymap.fits.gz','bayestar.fits.gz','BW_skymap.fits','LIB_skymap.fits.gz','skyprobcc_cWB.fits']
    #fileorder = ['LALInference3d.fits.gz','bayestar3d.fits.gz','bayestar.fits.gz']

    fileorder = ['LALInference_skymap.fits.gz','bayestar.fits.gz','BW_skymap.fits','LIB_skymap.fits.gz','skyprobcc_cWB.fits']

    for event in events:
        eventinfo = {}
        for key in keys:
            if not key in event: continue
            eventinfo[key] = event[key]
        eventinfo['gpstime'] = float(eventinfo['gpstime'])
        if eventinfo['far'] == None:
            eventinfo['far'] = np.nan

        #for key in event:
        #    print key
 
        #if not eventinfo['graceid'] == "G211117": continue
 
        triggerfile = "%s/%s.txt"%(triggerDir,eventinfo['graceid'])
        #skymapfile = '%s/%s.fits.gz'%(skymapDir,eventinfo['graceid'])
        skymapfile = '%s/%s.fits'%(skymapDir,eventinfo['graceid'])
        if os.path.isfile(triggerfile) and os.path.isfile(skymapfile):
            print "Already have info for %s... continuing."%event["graceid"]
            continue
        print "Getting info for %s"%event["graceid"]

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
            print "Looking for EM bright file..."
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
             print "No EM bright file..."            

        try:
            print "Looking for cWB file..."
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
            print "No cWB file..."
    
        #if len(gpss) > 0:
        #    dist = np.absolute(eventinfo['gpstime'] - np.array(gpss))
        #    if np.min(dist) < 5:
        #        continue
        #gpss.append(eventinfo['gpstime'])
    
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
            print "Download of skymaps file for %s failed..."%eventinfo['graceid']
        else:
   
            skymap = open(skymapfile,'w')
            skymap.write(r.read())
            skymap.close()
       
            print skymapfile 
            prob = hp.read_map(skymapfile)
            try:
                prob, distmu, distsigma, distnorm = hp.read_map(skymapfile,field=(0,1,2,3)) 
                distanceInfo = True
            except:
                distanceInfo = False
                print "No distance information"
            ra, dec, dist = opts.ra, opts.dec, opts.dist
            theta = 0.5 * np.pi - np.deg2rad(dec)
            phi = np.deg2rad(ra)

            npix = len(prob)
            nside = hp.npix2nside(npix)

            if distanceInfo:
                ipix = hp.ang2pix(nside, theta, phi)
                pixarea = hp.nside2pixarea(nside)
                pixarea_deg2 = hp.nside2pixarea(nside, degrees=True)

                dp_dA = prob[ipix] / pixarea
                dp_dA_deg2 = prob[ipix] / pixarea_deg2
     
                r = np.linspace(0, 1500)
                dp_dr = r**2 * distnorm[ipix] * norm(distmu[ipix], distsigma[ipix]).pdf(r)
                dp_dV = prob[ipix] * distnorm[ipix] * norm(distmu[ipix], distsigma[ipix]).pdf(dist) / pixarea
                dp_dr_sky = [np.sum(prob*rr**2 * distnorm * norm(distmu, distsigma).pdf(rr)) for rr in r]
                dp_dr_norm = np.cumsum(dp_dr / np.sum(dp_dr))
                dp_dr_sky_norm = np.cumsum(dp_dr_sky / np.sum(dp_dr_sky))
 
                dp_dr_norm_10 = np.argmin(np.abs(dp_dr_norm - 0.1))
                dp_dr_norm_90 = np.argmin(np.abs(dp_dr_norm - 0.9))
                dp_dr_sky_norm_10 = np.argmin(np.abs(dp_dr_sky_norm - 0.1))
                dp_dr_sky_norm_90 = np.argmin(np.abs(dp_dr_sky_norm - 0.9))

                print "Probability per steradian: %.5e"%dp_dA 
                print "Probability per deg^2: %.5e"%dp_dA_deg2
                print "Probability per unit volume: %.5e / Mpc^3"%dp_dV
                print "Line-of-sight 10-90 (mpc): %.0f - %.0f"%(r[dp_dr_norm_10],r[dp_dr_norm_90])
                print "All-sky 10-90 (mpc): %.0f - %.0f"%(r[dp_dr_sky_norm_10],r[dp_dr_sky_norm_90])

            index = np.argmax(prob)
            theta, phi = hp.pix2ang(nside, index)
            ra = phi
            dec = 0.5*np.pi - theta
        
            ra = (ra / (2*np.pi)) * 24.0
            dec = dec * (360.0/(2*np.pi))
      
            thisplotDir = os.path.join(plotDir,eventinfo['graceid'])
            if not os.path.isdir(thisplotDir):
                os.mkdir(thisplotDir)
        
            plotName = os.path.join(thisplotDir,'mollview.png')
            hp.mollview(prob)
            plt.show()
            plt.savefig(plotName,dpi=200)
            plt.close('all')
   
            if distanceInfo:
                plotName = os.path.join(thisplotDir,'prob_dist.png')
                plt.figure(figsize=(8, 6))
                plt.plot(r, dp_dr)
                plt.xlabel('distance (Mpc)')
                plt.ylabel('prob Mpc$^{-1}$')
                plt.show()
                plt.savefig(plotName,dpi=200)
                plt.close('all')
 
                plotName = os.path.join(thisplotDir,'prob_dist_sky.png')
                plt.figure(figsize=(8, 6))
                plt.plot(r, dp_dr_sky)
                plt.plot([dist,dist],[0,np.max(dp_dr_sky)],'r--')
                plt.xlabel('distance (Mpc)')
                plt.ylabel('prob Mpc$^{-1}$')
                plt.show()
                plt.savefig(plotName,dpi=200)
                plt.close('all')
 
        f = open(triggerfile,"w")
        print "%s %.0f %.10f %.10f %.10f %.10f %.10f %.10f"%(eventinfo['graceid'], eventinfo['gpstime'], eventinfo['far'], ra, dec, mjds[0], mjds[1],timediff)
        f.write("%.0f %.10f %.10f %.10f %.10f %.10f\n"%(eventinfo['gpstime'],ra,dec,mjds[0],mjds[1],timediff))
        f.close()

opts = parse_commandline()
 
triggerDir = os.path.join(opts.outputDir,"triggers")
skymapDir = os.path.join(opts.outputDir,"skymaps")
plotDir = os.path.join(opts.outputDir,"plots")
 
if not os.path.isdir(opts.outputDir): os.mkdir(opts.outputDir)
if not os.path.isdir(triggerDir): os.mkdir(triggerDir)
if not os.path.isdir(skymapDir): os.mkdir(skymapDir)
if not os.path.isdir(plotDir): os.mkdir(plotDir)

rm_command = "rm skymap.fits.gz*"
os.system(rm_command)

# Instantiate client
g = GraceDb()
#g = GraceDb(url)
#g = GraceDbBasic()
#g = GraceDbBasic(url)

eventString = get_eventstring(opts)
if opts.doGPSLoop:
    opts.doGPS = True
    while True:
        endtime = Time(datetime.utcnow(), scale='utc')
        starttime = t = Time(endtime.gps - 86400.0, format='gps', scale='utc')
        opts.startGPS = starttime.gps
        opts.endGPS = endtime.gps

        eventString = get_eventstring(opts)
        # REST API returns an iterator
        events = g.events('%s'%eventString)

        download_events(events)
else:
    eventString = get_eventstring(opts)
    print eventString
    # REST API returns an iterator
    events = g.events('%s'%eventString)

    download_events(events) 

