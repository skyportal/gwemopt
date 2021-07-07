
import os, sys
import time
import copy
import numpy as np
import healpy as hp
import itertools
import glob
import astropy, astroplan

from scipy.stats import norm
from scipy.optimize import minimize

import requests
import urllib.parse
import re

import astropy.coordinates
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy import table

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.path

import ligo.segments as segments
import ligo.skymap.distance as ligodist
from ligo.skymap.io import read_sky_map
from ligo.skymap.bayestar import rasterize

import gwemopt.moc
import gwemopt.tiles
import gwemopt.segments
import gwemopt.scheduler

def readParamsFromFile(file):
    """@read gwemopt params file

    @param file
        gwemopt params file
    """

    params = {}
    if os.path.isfile(file):
        with open(file,'r') as f:
            for line in f:
                line_without_return = line.split("\n")
                line_split = line_without_return[0].split(" ")
                line_split = list(filter(None, line_split))
                if line_split:
                    try:
                        params[line_split[0]] = float(line_split[1])
                    except:
                        params[line_split[0]] = line_split[1]
    return params

def params_checker(params):
    """"Assigns defaults to params."""
    do_Parameters = ["do3D","doEvent","doSuperSched","doMovie_supersched","doSkymap","doSamples","doCoverage","doSchedule","doPlots","doDatabase","doMovie","doTiles","doIterativeTiling","doMinimalTiling","doOverlappingScheduling","doPerturbativeTiling","doOrderByObservability","doCatalog","doUseCatalog","doCatalogDatabase","doObservability","doSkybrightness","doEfficiency","doCalcTiles","doTransients","doSingleExposure","doAlternatingFilters","doMaxTiles","doReferences","doChipGaps","doUsePrimary","doUseSecondary","doSplit","doParallel","writeCatalog","doFootprint","doBalanceExposure","doBlocks","doUpdateScheduler","doTreasureMap","doRASlice","doRASlices","doRotate","doMindifFilt","doTrueLocation","doAvoidGalacticPlane"]
 
    for parameter in do_Parameters:
        if parameter not in params.keys():
            params[parameter] = False

    if "skymap" not in params.keys():
        params["skymap"] = '../output/skymaps/G268556.fits'

    if "gpstime" not in params.keys():
        params["gpstime"] = 1167559936.0

    if "outputDir" not in params.keys():
        params["outputDir"] = "../output"

    if "tilingDir" not in params.keys():
        params["tilingDir"] = "../tiling"

    if "catalogDir" not in params.keys():
        params["catalogDir"] = "../catalogs"

    if "event" not in params.keys():
        params["event"] = "G268556"

    if "coverageFiles" not in params.keys():
        params["coverageFiles"] = "../data/ATLAS_GW170104.dat"

    if "telescopes" not in params.keys():
        params["telescopes"] = 'ATLAS'

    if type(params["telescopes"]) == str:
        params["telescopes"] = params["telescopes"].split(",")

    if "lightcurveFiles" not in params.keys():
        params["lightcurveFiles"] = "../lightcurves/Me2017_H4M050V20.dat"

    if "tilesType" not in params.keys():
        params["tilesType"] = "moc"

    if "scheduleType" not in params.keys():
        params["scheduleType"] = "greedy"

    if "timeallocationType" not in params.keys():
        params["timeallocationType"] = "powerlaw"

    if "Ninj" not in params.keys():
        params["Ninj"] = 1000

    if "Ndet" not in params.keys():
        params["Ndet"] = 1
    
    if "Ntiles" not in params.keys():
        params["Ntiles"] = 10

    if "Ntiles_cr" not in params.keys():
        params["Ntiles_cr"] = 0.70

    if "Dscale" not in params.keys():
        params["Dscale"] = 1.0

    if "nside" not in params.keys():
        params["nside"] = 256

    if "Tobs" not in params.keys():
        params["Tobs"] = np.array([0.0,1.0])

    if "powerlaw_cl" not in params.keys():
        params["powerlaw_cl"] = 0.9

    if "powerlaw_n" not in params.keys():
        params["powerlaw_n"] = 1.0

    if "powerlaw_dist_exp" not in params.keys():
        params["powerlaw_dist_exp"] = 0

    if "galaxies_FoV_sep" not in params.keys():
        params["galaxies_FoV_sep"] = 1.0

    if "footprint_ra" not in params.keys():
        params["footprint_ra"] = 30.0

    if "footprint_dec" not in params.keys():
        params["footprint_dec"] = 60.0

    if "footprint_radius" not in params.keys():
        params["footprint_radius"] = 10.0

    if "transientsFile" not in params.keys():
        params["transientsFile"] = "../data/GW190425/transients.dat"

    if "transients_to_catalog" not in params.keys():
        params["transients_to_catalog"] = 0.8

    if "dt" not in params.keys():
        params["dt"] = 14.0

    if "galaxy_catalog" not in params.keys():
        params["galaxy_catalog"] = "GLADE"
    
    if "filters" not in params.keys():
        params["filters"] = ['r','g','r']

    if "exposuretimes" not in params.keys():
        params["exposuretimes"] = np.array([30.0,30.0,30.0])

    if "max_nb_tiles" not in params.keys():
        params["max_nb_tiles"] = np.array([-1,-1,-1])

    if "mindiff" not in params.keys():
        params["mindiff"] = 0.0

    if "airmass" not in params.keys():
        params["airmass"] = 2.5

    if "iterativeOverlap" not in params.keys():
        params["iterativeOverlap"] = 0.0

    if "maximumOverlap" not in params.keys():
        params["maximumOverlap"] = 1.0

    if "catalog_n" not in params.keys():
        params["catalog_n"] = 1.0

    if "galaxy_grade" not in params.keys():
        params["galaxy_grade"] = "S"

    if "AGN_flag" not in params.keys():
        params["AGN_flag"] = False

    if "splitType" not in params.keys():
        params["splitType"] = "regional"

    if "Nregions" not in params.keys():
        params["Nregions"] = 768

    if "configDirectory" not in params.keys():
        params["configDirectory"] = "../config/"

    if "Ncores" not in params.keys():
        params["Ncores"] = 4

    if "Nblocks" not in params.keys():
        params["Nblocks"] = 4

    if "unbalanced_tiles" not in params.keys():
        params["unbalanced_tiles"] = None

    if "treasuremap_token" not in params.keys():
        params["treasuremap_token"] = ""

    if "treasuremap_status" not in params.keys():
        params["treasuremap_status"] = ["planned","completed"]

    if "graceid" not in params.keys():
        params["graceid"] = "S190426c"

    if "raslice" not in params.keys():
        params["raslice"] = [0.0,24.0]

    if "nside_down" not in params.keys():
        params["nside_down"] = 2

    if "max_filter_sets" not in params.keys():
        params["max_filter_sets"] = 4

    if "absmag" not in params.keys():
        params["absmag"] = -15

    if "phi" not in params.keys():
        params["phi"] = 0

    if "theta" not in params.keys():
        params["theta"] = 0

    if "program_id" not in params.keys():
        params["program_id"] = -1

    if "galactic_limit" not in params.keys():
        params["galactic_limit"] = 15.0

    if "true_ra" not in params.keys():
        params["true_ra"] = 30.0
    if "true_dec" not in params.keys():
        params["true_dec"] = 60.0
    if "true_distance" not in params.keys():
        params["true_distance"] = 100.0

    if "config" not in params.keys():
        params["config"] = {}
        configFiles = glob.glob("%s/*.config"%params["configDirectory"])
        for configFile in configFiles:
            telescope = configFile.split("/")[-1].replace(".config","")
            if not telescope in params["telescopes"]: continue
            params["config"][telescope] = gwemopt.utils.readParamsFromFile(configFile)
            params["config"][telescope]["telescope"] = telescope
            if params["doSingleExposure"]:
                exposuretime = np.array(opts.exposuretimes.split(","),dtype=np.float)[0]
           
                nmag = -2.5*np.log10(np.sqrt(params["config"][telescope]["exposuretime"]/exposuretime))
                params["config"][telescope]["magnitude"] = params["config"][telescope]["magnitude"] + nmag
                params["config"][telescope]["exposuretime"] = exposuretime
            if "tesselationFile" in params["config"][telescope]:
                if not os.path.isfile(params["config"][telescope]["tesselationFile"]):
                    if params["config"][telescope]["FOV_type"] == "circle":
                        gwemopt.tiles.tesselation_spiral(params["config"][telescope])
                    elif params["config"][telescope]["FOV_type"] == "square":
                        gwemopt.tiles.tesselation_packing(params["config"][telescope])
                if params["tilesType"] == "galaxy":
                    params["config"][telescope]["tesselation"] = np.empty((3,))
                else:
                    params["config"][telescope]["tesselation"] = np.loadtxt(params["config"][telescope]["tesselationFile"],usecols=(0,1,2),comments='%')

            if "referenceFile" in params["config"][telescope]:
                refs = table.unique(table.Table.read(
                    params["config"][telescope]["referenceFile"],
                    format='ascii', data_start=2, data_end=-1)['field', 'fid'])
                reference_images =\
                    {group[0]['field']: group['fid'].astype(int).tolist()
                    for group in refs.group_by('field').groups}
                reference_images_map = {1: 'g', 2: 'r', 3: 'i'}
                for key in reference_images:
                    reference_images[key] = [reference_images_map.get(n, n)
                                             for n in reference_images[key]]
                params["config"][telescope]["reference_images"] = reference_images
                                                                     
            location = astropy.coordinates.EarthLocation(params["config"][telescope]["longitude"],params["config"][telescope]["latitude"],params["config"][telescope]["elevation"])
            observer = astroplan.Observer(location=location)
            params["config"][telescope]["observer"] = observer

    return params

def auto_rasplit(params,map_struct,nside_down):

    if params["do3D"]:
        prob_down, distmu_down,\
        distsigma_down, distnorm_down = ligodist.ud_grade(map_struct["prob"],\
                                                      map_struct["distmu"],\
                                                      map_struct["distsigma"],\
                                                      nside_down)
    else:
        prob_down = hp.ud_grade(map_struct["prob"],nside_down,power=-2) 

    npix = hp.nside2npix(nside_down)
    theta, phi = hp.pix2ang(nside_down, np.arange(npix))
    ra = np.rad2deg(phi)*24.0/360.0
    dec = np.rad2deg(0.5*np.pi - theta)

    ra_unique = np.unique(ra)
    prob_sum = np.zeros(ra_unique.shape)
    for ii, r in enumerate(ra_unique):
        idx = np.where(r == ra)[0]
        prob_sum[ii] = np.sum(prob_down[idx])

    sort_idx = np.argsort(prob_sum)[::-1]
    csm = np.empty(len(prob_sum))
    csm[sort_idx] = np.cumsum(prob_sum[sort_idx])

    idx = np.where(csm <= params["powerlaw_cl"])[0] 

    if (0 in idx) and (len(ra_unique)-1 in idx):
        wrap = True
    else:
        wrap = False

    dr = ra_unique[1] - ra_unique[0]
    segmentlist = segments.segmentlist()
    for ii in idx:
        ra1, ra2 = ra_unique[ii], ra_unique[ii]+dr
        segment = segments.segment(ra1, ra2)
        segmentlist = segmentlist + segments.segmentlist([segment])
        segmentlist.coalesce()

    if wrap:
        idxremove = []
        for ii, seg in enumerate(segmentlist):
            if np.isclose(seg[0], 0.0) and wrap:
                seg1 = seg
                idxremove.append(ii)
                continue
            if np.isclose(seg[1], 24.0) and wrap:
                seg2 = seg
                idxremove.append(ii)
                continue
       
        for ele in sorted(idxremove, reverse = True):  
            del segmentlist[ele] 
        
        raslices = []
        for segment in segmentlist:
            raslices.append([segment[0],segment[1]])
        raslices.append([seg2[0], seg1[1]])
    else:
        raslices = []
        for segment in segmentlist:
            raslices.append([segment[0],segment[1]])

    return raslices

def rotate_map(hmap, rot_theta, rot_phi):
    """
    Take hmap (a healpix map array) and return another healpix map array
    which is ordered such that it has been rotated in (theta, phi) by the
    amounts given.
    """
    nside = hp.npix2nside(len(hmap))

    # Get theta, phi for non-rotated map
    t,p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside))) #theta, phi

    # Define a rotator
    r = hp.Rotator(deg=False, rot=[rot_phi,rot_theta])

    # Get theta, phi under rotated co-ordinates
    trot, prot = r(t,p)

    # Interpolate map onto these co-ordinates
    rot_map = hp.get_interp_val(hmap, trot, prot)

    return rot_map

def read_skymap(params,is3D=False,map_struct=None):

    header = []
    if map_struct is None:
        map_struct = {}

        if params["doDatabase"]:
            models = params["models"]
            localizations_all = models.Localization.query.all()
            localizations = models.Localization.query.filter_by(dateobs=params["dateobs"],localization_name=params["localization_name"]).all()
            if localizations == None:
                raise ValueError("No localization with dateobs=%s"%params["dateobs"])
            else:
                prob_data = localizations[0].healpix
                prob_data = prob_data / np.sum(prob_data)
                map_struct["prob"] = prob_data
    
                distmu = localizations[0].distmu
                distsigma = localizations[0].distsigma
                distnorm = localizations[0].distnorm
    
                if distmu is None:            
                    map_struct["distmu"] = None
                    map_struct["distsigma"] = None
                    map_struct["distnorm"] = None
                else:
                    map_struct["distmu"] = np.array(distmu)
                    map_struct["distsigma"] = np.array(distsigma)
                    map_struct["distnorm"] = np.array(distnorm)
                    is3D = True
        else:
            filename = params["skymap"]
        
            if is3D:
                try:
                    healpix_data, header = hp.read_map(filename, field=(0,1,2,3), verbose=False,h=True)
                except:
                    table = read_sky_map(filename, moc=True, distances=True)
                    order = hp.nside2order(params["nside"])
                    t = rasterize(table, order)
                    result = t['PROB'], t['DISTMU'], t['DISTSIGMA'], t['DISTNORM']
                    healpix_data = hp.reorder(result, 'NESTED', 'RING')

                distmu_data = healpix_data[1]
                distsigma_data = healpix_data[2]
                prob_data = healpix_data[0]
                norm_data = healpix_data[3]

                map_struct["distmu"] = distmu_data / params["DScale"]
                map_struct["distsigma"] = distsigma_data / params["DScale"]
                map_struct["prob"] = prob_data
                map_struct["distnorm"] = norm_data
    
            else:
                prob_data, header = hp.read_map(filename, field=0, verbose=False,h=True)
                prob_data = prob_data / np.sum(prob_data)
        
                map_struct["prob"] = prob_data

    if params["doRotate"]:
        for key in map_struct.keys():
            map_struct[key] = rotate_map(map_struct[key], np.deg2rad(params["theta"]), np.deg2rad(params["phi"]))
        map_struct["prob"] = map_struct["prob"] / np.sum(map_struct["prob"])
 
    natural_nside = hp.pixelfunc.get_nside(map_struct["prob"])
    nside = params["nside"]
    
    print("natural_nside =", natural_nside)
    print("nside =", nside)
    
    if not is3D:
        map_struct["prob"] = hp.ud_grade(map_struct["prob"],nside,power=-2)

    if is3D:
        if natural_nside != nside:
            
            map_struct["prob"] = hp.pixelfunc.ud_grade(map_struct["prob"],nside,power=-2)
            map_struct["distmu"] = hp.pixelfunc.ud_grade(map_struct["distmu"],nside)
            map_struct["distsigma"] = hp.pixelfunc.ud_grade(map_struct["distsigma"],nside)
            map_struct["distnorm"] = hp.pixelfunc.ud_grade(map_struct["distnorm"],nside)

            map_struct["distmu"][map_struct["distmu"] < -1e+30] = np.inf

        nside_down = 32

        distmu_down = hp.pixelfunc.ud_grade(map_struct["distmu"],nside_down)
        distsigma_down = hp.pixelfunc.ud_grade(map_struct["distsigma"],nside_down)
        distnorm_down = hp.pixelfunc.ud_grade(map_struct["distnorm"],nside_down)

        map_struct["distmed"], map_struct["diststd"], mom_norm = ligodist.parameters_to_moments(
                                                                          map_struct["distmu"],
                                                                        map_struct["distsigma"])

        distmu_down[distmu_down < -1e+30] = np.inf

        map_struct["distmed"] = hp.ud_grade(map_struct["distmed"],nside,power=-2)
        map_struct["diststd"] = hp.ud_grade(map_struct["diststd"],nside,power=-2)

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5*np.pi - theta)

    map_struct["ra"] = ra
    map_struct["dec"] = dec

    if params["doRASlice"]:
        ra_low, ra_high = params["raslice"][0], params["raslice"][1]
        if ra_low <= ra_high:
            ipix = np.where((ra_high*360.0/24.0 < ra) | (ra_low*360.0/24.0 > ra))[0]
        else:
            ipix = np.where((ra_high*360.0/24.0 < ra) & (ra_low*360.0/24.0 > ra))[0]
        map_struct["prob"][ipix] = 0.0
        map_struct["prob"] = map_struct["prob"] / np.sum(map_struct["prob"])

    if params["doAvoidGalacticPlane"]:
        coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
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
    
    for j in range(len(header)):
        if header[j][0] == "DATE":
            map_struct["trigtime"] = header[j][1]

    return map_struct   

def samples_from_skymap(map_struct, is3D = False, Nsamples = 100):

    prob_data_sorted = np.sort(map_struct["prob"])[::-1]
    prob_data_indexes = np.argsort(map_struct["prob"])[::-1]
    prob_data_cumsum = np.cumsum(prob_data_sorted)

    rand_values = np.random.rand(Nsamples,)

    ras = []
    decs = []
    dists = []

    if is3D:
        r = np.linspace(0, 2000)
        rand_values_dist = np.random.rand(Nsamples,)

    for ii in range(Nsamples):
        ipix = np.argmin(np.abs(prob_data_cumsum-rand_values[ii]))
        ra_inj = map_struct["ra"][prob_data_indexes][ipix]
        dec_inj = map_struct["dec"][prob_data_indexes][ipix]

        ras.append(ra_inj)
        decs.append(dec_inj)    

        if is3D:
            dp_dr = r**2 * map_struct["distnorm"][prob_data_indexes][ipix] * norm(map_struct["distmu"][prob_data_indexes][ipix], map_struct["distsigma"][prob_data_indexes][ipix]).pdf(r)
            dp_dr_norm = np.cumsum(dp_dr / np.sum(dp_dr))
            idx = np.argmin(np.abs(dp_dr_norm-rand_values_dist[ii]))
            dist_inj = r[idx]
            dists.append(dist_inj)
        else:
            dists.append(50.0)

    samples_struct = {}
    samples_struct["ra"] = np.array(ras)
    samples_struct["dec"] = np.array(decs)
    samples_struct["dist"] = np.array(dists)

    return samples_struct

def get_ellipse_coords(a=0.0, b=0.0, x=0.0, y=0.0, angle=0.0, npts=10):
    """ Draws an ellipse using (360*k + 1) discrete points; based on pseudo code
    given at http://en.wikipedia.org/wiki/Ellipse
    k = 1 means 361 points (degree by degree)
    a = major axis distance,
    b = minor axis distance,
    x = offset along the x-axis
    y = offset along the y-axis
    angle = clockwise rotation [in degrees] of the ellipse;
        * angle=0  : the ellipse is aligned with the positive x-axis
        * angle=30 : rotated 30 degrees clockwise from positive x-axis
    """
    pts = np.zeros((npts, 2))

    beta = -angle * np.pi/180.0
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    alpha = np.linspace(0,2*np.pi,npts) 

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    
    pts[:, 0] = x + (a * cos_alpha * cos_beta - b * sin_alpha * sin_beta)
    pts[:, 1] = y + (a * cos_alpha * sin_beta + b * sin_alpha * cos_beta)

    return pts

def getCirclePixels(ra_pointing, dec_pointing, radius, nside, alpha=0.4, color='k', edgecolor='k', rotation=None):

    theta = 0.5 * np.pi - np.deg2rad(dec_pointing)
    phi = np.deg2rad(ra_pointing)

    xyz = hp.ang2vec(theta, phi)
    ipix = hp.query_disc(nside, xyz, np.deg2rad(radius))

    radecs = get_ellipse_coords(a=radius/np.cos(np.deg2rad(dec_pointing)), b=radius, x=ra_pointing, y=dec_pointing, angle=0.0, npts=25)
    idx = np.where(radecs[:,1] > 90.0)[0]
    radecs[idx,1] = 180.0 - radecs[idx,1] 
    idx = np.where(radecs[:,1] < -90.0)[0]
    radecs[idx,1] = -180.0 - radecs[idx,1]    
    idx = np.where(radecs[:,0] > 360.0)[0]
    radecs[idx,0] = 720.0 - radecs[idx,0]
    idx = np.where(radecs[:,0] < 0.0)[0]
    radecs[idx,0] = 360.0 + radecs[idx,0]

    radecs = np.array(radecs)
    idx1 = np.where(radecs[:,0]>=180.0)[0]
    idx2 = np.where(radecs[:,0]<180.0)[0]
    idx3 = np.where(radecs[:,0]>300.0)[0]
    idx4 = np.where(radecs[:,0]<60.0)[0]
    if (len(idx1)>0 and len(idx2)>0) and not (len(idx3)>0 and len(idx4)>0):
        alpha = 0.0

    xyz = hp.ang2vec(radecs[:,0],radecs[:,1],lonlat=True)

    proj = hp.projector.MollweideProj(rot=rotation, coord=None)
    x,y = proj.vec2xy(xyz[:,0],xyz[:,1],xyz[:,2])
    xy = np.zeros(radecs.shape)
    xy[:,0] = x
    xy[:,1] = y
    #path = matplotlib.path.Path(xyz[:,1:3])
    path = matplotlib.path.Path(xy)
    patch = matplotlib.patches.PathPatch(path, alpha=alpha, color=color, fill=True, zorder=3, edgecolor=edgecolor)

    area = np.pi * radius**2

    return ipix, radecs, patch, area

def getRectanglePixels(ra_pointing, dec_pointing, raSide, decSide, nside, alpha = 0.4, color='k', edgecolor='k', rotation=None):

    area = raSide*decSide

    decCorners = (dec_pointing - decSide / 2.0, dec_pointing + decSide / 2.0)

    #security for the periodic limit conditions 
    radecs = []
    for d in decCorners:
        if d > 90.:
            d = 180. - d
        elif d < -90.:
            d = -180 - d

        raCorners = (ra_pointing - (raSide / 2.0) / np.cos(np.deg2rad(d)) , ra_pointing + (raSide / 2.0) / np.cos(np.deg2rad(d)))
        #security for the periodic limit conditions 
        for r in raCorners:
            if r > 360.:
                r = r - 360.
            elif r < 0.:
                r = 360. + r
            radecs.append([r,d])

    radecs = np.array(radecs)
    idx1 = np.where(radecs[:,0]>=180.0)[0]
    idx2 = np.where(radecs[:,0]<180.0)[0]
    idx3 = np.where(radecs[:,0]>300.0)[0]
    idx4 = np.where(radecs[:,0]<60.0)[0]
    if (len(idx1)>0 and len(idx2)>0) and not (len(idx3)>0 and len(idx4)>0):
        alpha = 0.0

    idx1 = np.where(np.abs(radecs[:,1])>=87.0)[0]
    if len(idx1) == 4:
        return [], [], [], []

    idx1 = np.where((radecs[:,1]>=87.0) | (radecs[:,1]<=-87.0))[0]
    if len(idx1)>0:
        radecs = np.delete(radecs, idx1[0], 0)

    xyz = []
    for r, d in radecs:
        xyz.append(hp.ang2vec(r, d, lonlat=True))

    npts, junk = radecs.shape
    if npts == 4:
        xyz = [xyz[0], xyz[1],xyz[3], xyz[2]]
        ipix = hp.query_polygon(nside, np.array(xyz))
    else:
        ipix = hp.query_polygon(nside, np.array(xyz))

    #idx1 = np.where((radecs[:,1]>=70.0) | (radecs[:,1]<=-70.0))[0]
    #idx2 = np.where((radecs[:,0]>300.0) | (radecs[:,0]<60.0))[0]
    #if (len(idx1) == 0) or (len(idx2) > 0):
    #    return [], [], [], []

    xyz = np.array(xyz)
    proj = hp.projector.MollweideProj(rot=rotation, coord=None)
    x,y = proj.vec2xy(xyz[:,0],xyz[:,1],xyz[:,2])
    xy = np.zeros(radecs.shape)
    xy[:,0] = x
    xy[:,1] = y
    path = matplotlib.path.Path(xy)
    patch = matplotlib.patches.PathPatch(path, alpha=alpha, color=color, fill=True, zorder=3, edgecolor=edgecolor)

    return ipix, radecs, patch, area

def getSquarePixels(ra_pointing, dec_pointing, tileSide, nside, alpha = 0.4, color='k', edgecolor='k', rotation=None):

    area = tileSide*tileSide

    decCorners = (dec_pointing - tileSide / 2.0, dec_pointing + tileSide / 2.0)

    #security for the periodic limit conditions 
    radecs = []
    for d in decCorners:
        if d > 90.:
            d = 180. - d
        elif d < -90.:
            d = -180 - d

        raCorners = (ra_pointing - (tileSide / 2.0) / np.cos(np.deg2rad(d)) , ra_pointing + (tileSide / 2.0) / np.cos(np.deg2rad(d)))

        #security for the periodic limit conditions 
        for r in raCorners:
            if r > 360.:
                r = r - 360.
            elif r < 0.:
                r = 360. + r
            radecs.append([r,d])

    radecs = np.array(radecs)
    idx1 = np.where(radecs[:,0]>=180.0)[0] 
    idx2 = np.where(radecs[:,0]<180.0)[0]
    idx3 = np.where(radecs[:,0]>300.0)[0]
    idx4 = np.where(radecs[:,0]<60.0)[0]
    if (len(idx1)>0 and len(idx2)>0) and not (len(idx3)>0 and len(idx4)>0):
        alpha = 0.0

    idx1 = np.where(np.abs(radecs[:,1])>=87.0)[0] 
    if len(idx1) == 4:
        return [], [], [], []

    idx1 = np.where((radecs[:,1]>=87.0) | (radecs[:,1]<=-87.0))[0]
    if len(idx1)>0:
        radecs = np.delete(radecs, idx1[0], 0)

    xyz = []
    for r, d in radecs:
        xyz.append(hp.ang2vec(r, d, lonlat=True))

    npts, junk = radecs.shape
    if npts == 4:
        xyz = [xyz[0], xyz[1],xyz[3], xyz[2]]
        ipix = hp.query_polygon(nside, np.array(xyz))
    else:
        ipix = hp.query_polygon(nside, np.array(xyz))

    #idx1 = np.where((radecs[:,1]>=70.0) | (radecs[:,1]<=-70.0))[0]
    #idx2 = np.where((radecs[:,0]>300.0) | (radecs[:,0]<60.0))[0]
    #if (len(idx1) == 0) or (len(idx2) > 0):
    #    return [], [], [], []

    xyz = np.array(xyz)
    proj = hp.projector.MollweideProj(rot=rotation, coord=None) 
    x,y = proj.vec2xy(xyz[:,0],xyz[:,1],xyz[:,2])
    xy = np.zeros(radecs.shape)
    xy[:,0] = x
    xy[:,1] = y
    path = matplotlib.path.Path(xy)
    patch = matplotlib.patches.PathPatch(path, alpha=alpha, color=color, fill=True, zorder=3, edgecolor=edgecolor)
    
    return ipix, radecs, patch, area

def integrationTime(T_obs, pValTiles, func=None, T_int=60.0):
    '''
    METHOD :: This method accepts the probability values of the ranked tiles, the 
              total observation time and the rank of the source tile. It returns 
              the array of time to be spent in each tile which is determined based
              on the localizaton probability of the tile. How the weight factor is 
              computed can also be supplied in functional form. Default is linear.
                      
    pValTiles :: The probability value of the ranked tiles. Obtained from ZTF_RT 
                             output
    T_obs     :: Total observation time available for the follow-up.
    func      :: functional form of the weight. Default is linear. 
                             For example, use x**2 to use a quadratic function.
    '''

    if func is None:
            f = lambda x: x
    else:
            f = lambda x: eval(func)
    fpValTiles = f(pValTiles)
    modified_prob = fpValTiles/np.sum(fpValTiles)
    modified_prob[np.isnan(modified_prob)] = 0.0
    t_tiles = modified_prob * T_obs ### Time spent in each tile if not constrained
    #t_tiles[t_tiles > 1200.0] = 1200.0 ### Upper limit of exposure time
    #t_tiles[t_tiles < 60] = 60.0 ### Lower limit of exposure time
    t_tiles = T_int*np.round(t_tiles/T_int)
    #Obs = np.cumsum(t_tiles) <= T_obs ### Tiles observable in T_obs seconds
    #time_per_tile = t_tiles[Obs] ### Actual time spent per tile

    return t_tiles

def observability(params, map_struct):

    airmass = params["airmass"]
    nside = params["nside"]
    npix = hp.nside2npix(nside)
    gpstime = params["gpstime"]
    event_time = Time(gpstime, format='gps', scale='utc')
    dts = np.arange(0,7,1.0/24.0)
    dts = np.arange(0,7,1.0/4.0)

    observatory_struct = {}

    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]

        observatory = astropy.coordinates.EarthLocation(
            lat=config_struct["latitude"]*u.deg, lon=config_struct["longitude"]*u.deg, height=config_struct["elevation"]*u.m)

        # Look up (celestial) spherical polar coordinates of HEALPix grid.
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        # Convert to RA, Dec.
        radecs = astropy.coordinates.SkyCoord(
            ra=phi*u.rad, dec=(0.5*np.pi - theta)*u.rad)

        observatory_struct[telescope] = {}
        observatory_struct[telescope]["prob"] = copy.deepcopy(map_struct["prob"])
        observatory_struct[telescope]["observability"] = np.zeros((npix,))
        observatory_struct[telescope]["dts"] = {}

        for dt in dts:
            time = event_time+TimeDelta(dt*u.day)

            # Alt/az reference frame at observatory, now
            frame = astropy.coordinates.AltAz(obstime=time, location=observatory)
            # Transform grid to alt/az coordinates at observatory, now
            altaz = radecs.transform_to(frame)

            # Where is the sun, now?
            sun_altaz = astropy.coordinates.get_sun(time).transform_to(altaz)

            # How likely is it that the (true, unknown) location of the source
            # is within the area that is visible, now? Demand that sun is at
            # least 18 degrees below the horizon and that the airmass
            # (secant of zenith angle approximation) is at most 2.5.
            idx = np.where((altaz.alt >= 30*u.deg) &  (sun_altaz.alt <= -18*u.deg) & (altaz.secz <= airmass))[0]
            observatory_struct[telescope]["dts"][dt] = np.zeros((npix,))
            observatory_struct[telescope]["dts"][dt][idx] = 1
            observatory_struct[telescope]["observability"][idx] = 1
        observatory_struct[telescope]["prob"] = observatory_struct[telescope]["prob"]*observatory_struct[telescope]["observability"]

    return observatory_struct

def get_exposures(params, config_struct, segmentlist):
    '''
    Convert the availability times to a list segments with the length of telescope exposures.
    segmentlist: the segments that the telescope can do the follow-up.
    '''
    exposurelist = segments.segmentlist()
    if "overhead_per_exposure" in config_struct.keys(): overhead = config_struct["overhead_per_exposure"]
    else: overhead = 0.0

    # add the filter change time to the total overheads for integrated
    if not params["doAlternatingFilters"]: overhead = overhead + config_struct["filt_change_time"]

    exposure_time = np.max(params["exposuretimes"])

    for ii in range(len(segmentlist)):
        start_segment, end_segment = segmentlist[ii][0], segmentlist[ii][1]
        exposures = np.arange(start_segment, end_segment, (overhead+exposure_time)/86400.0)

        for jj in range(len(exposures)):
            exposurelist.append(segments.segment(exposures[jj],exposures[jj]+exposure_time/86400.0))

    return exposurelist

def perturb_tiles(params, config_struct, telescope, map_struct, tile_struct):

    map_struct_hold = copy.deepcopy(map_struct)
    ipix_keep = map_struct_hold["ipix_keep"]
    nside = params["nside"]

    if config_struct["FOV_type"] == "square":
        width = config_struct["FOV"]*0.5
    elif config_struct["FOV_type"] == "circle":
        width = config_struct["FOV"]*1.0

    moc_struct = {}
    keys = list(tile_struct.keys())
    for ii, key in enumerate(keys):
        if tile_struct[key]['prob'] == 0.0: continue

        if np.mod(ii,100) == 0:
            print("Optimizing tile %d/%d" % (ii, len(keys)))

        x0 = [tile_struct[key]["ra"], tile_struct[key]["dec"]]
        FOV = config_struct["FOV"]
        bounds = [[tile_struct[key]["ra"]-width, tile_struct[key]["ra"]+width],
                  [tile_struct[key]["dec"]-width, tile_struct[key]["dec"]+width]]

        ras = np.linspace(tile_struct[key]["ra"]-width, tile_struct[key]["ra"]+width, 5)
        decs = np.linspace(tile_struct[key]["dec"]-width, tile_struct[key]["dec"]+width, 5)
        RAs, DECs = np.meshgrid(ras, decs)
        ras, decs = RAs.flatten(), DECs.flatten()

        vals = []
        for ra, dec in zip(ras, decs):
            if np.abs(dec) > 90:
                vals.append(0)
                continue

            moc_struct_temp = gwemopt.moc.Fov2Moc(params, config_struct, telescope, ra, dec, nside)
            idx = np.where(map_struct_hold["prob"][moc_struct_temp["ipix"]] == -1)[0]
            idx = np.setdiff1d(idx,ipix_keep)
            if len(map_struct_hold["prob"][moc_struct_temp["ipix"]]) == 0:
                rat = 0.0
            else:
                rat = float(len(idx)) / float(len(map_struct_hold["prob"][moc_struct_temp["ipix"]]))
            if rat > params["maximumOverlap"]:
                val = 0.0
            else:
                ipix = moc_struct_temp["ipix"]
                if len(ipix) == 0:
                    val = 0.0
                else: 
                    vals_to_sum = map_struct_hold["prob"][ipix]
                    vals_to_sum[vals_to_sum < 0] = 0
                    val = np.sum(vals_to_sum) 
            vals.append(val)
        idx = np.argmax(vals)
        ra, dec = ras[idx], decs[idx]
        moc_struct[key] = gwemopt.moc.Fov2Moc(params, config_struct, telescope, ra, dec, nside)

        map_struct_hold['prob'][moc_struct[key]["ipix"]] = -1
        ipix_keep = np.setdiff1d(ipix_keep, moc_struct[key]["ipix"])

    if params["timeallocationType"] == "absmag":
        tile_struct = gwemopt.tiles.absmag_tiles_struct(params, config_struct, telescope, map_struct, moc_struct)
    elif params["timeallocationType"] == "powerlaw":
        tile_struct = gwemopt.tiles.powerlaw_tiles_struct(params, config_struct, telescope, map_struct, moc_struct)
    tile_struct = gwemopt.segments.get_segments_tiles(params, config_struct, tile_struct)
 
    return tile_struct

def slice_map_tiles(params, map_struct, coverage_struct):

    prob = copy.deepcopy(map_struct["prob"])
    prob[prob < 0] = 0.0

    sort_idx = np.argsort(prob)[::-1]
    csm = np.empty(len(prob))
    csm[sort_idx] = np.cumsum(prob[sort_idx])
    ipix_keep = np.where(csm <= params["iterativeOverlap"])[0]

    for ii in range(len(coverage_struct["ipix"])):
        data = coverage_struct["data"][ii,:]
        filt = coverage_struct["filters"][ii]
        ipix = coverage_struct["ipix"][ii]
        patch = coverage_struct["patch"][ii]
        FOV = coverage_struct["FOV"][ii]
        area = coverage_struct["area"][ii]

        observ_time, exposure_time, field_id, prob, airmass = data[2], data[4], data[5], data[6], data[7]

        ipix_slice = np.setdiff1d(ipix, ipix_keep)
        if len(ipix_slice) == 0: continue
        map_struct["prob"][ipix_slice] = -1

    return map_struct

def slice_number_tiles(params, telescope, tile_struct, coverage_struct):

    idx = params["telescopes"].index(telescope)
    max_nb_tile = params["max_nb_tiles"][idx]
    if max_nb_tile < 0:
        return tile_struct, False

    keys = tile_struct.keys()
    keys_scheduled = np.unique(coverage_struct["data"][:,5])
    
    if len(keys_scheduled) <= max_nb_tile:
        return tile_struct, False

    prob = np.zeros((len(keys),))
    for ii, key in enumerate(keys):
        if key in keys_scheduled:
            prob[ii] = prob[ii] + tile_struct[key]['prob']

    sort_idx = np.argsort(prob)[::-1]
    idx_keep = sort_idx[:int(max_nb_tile)]

    for ii, key in enumerate(keys):
        # in the golden tile set
        if ii in idx_keep: continue
        tile_struct[key]['prob'] = 0.0

    return tile_struct, True

def eject_tiles(params, telescope, tile_struct):
    
    for field_id in params["scheduled_fields"][telescope]:
        tile_struct[field_id]['prob'] = 0.0
    
    return tile_struct

def balance_tiles(params, tile_struct, coverage_struct):

    filters, exposuretimes = params["filters"], params["exposuretimes"]

    keys_scheduled = coverage_struct["data"][:,5]

    doReschedule = False
    unique, freq = np.unique(keys_scheduled, return_counts=True)
    balanced_fields = np.sum(freq==len(filters))

    params["unbalanced_tiles"] = [key for i,key in enumerate(unique) if freq[i] != len(filters)]

    if len(params["unbalanced_tiles"]) != 0:
        doReschedule = True

    return doReschedule, balanced_fields

def erase_unbalanced_tiles(params,coverage_struct):

    idxs = np.isin(coverage_struct["data"][:,5],params["unbalanced_tiles"])
    out = np.nonzero(idxs)[0]
    
    coverage_struct["data"] = np.delete(coverage_struct["data"],out,axis=0)
    coverage_struct["filters"] = np.delete(coverage_struct["filters"],out)
    coverage_struct["area"] = np.delete(coverage_struct["area"],out)
    coverage_struct["FOV"] = np.delete(coverage_struct["FOV"],out)
    coverage_struct["telescope"] = np.delete(coverage_struct["telescope"],out)
    
    for i in out[::-1]:
        del coverage_struct["patch"][i]
        del coverage_struct["ipix"][i]

    return coverage_struct


def slice_galaxy_tiles(params, tile_struct, coverage_struct):

    coverage_ras = coverage_struct["data"][:,0]
    coverage_decs = coverage_struct["data"][:,1]

    if len(coverage_ras) == 0:
        return tile_struct

    keys = tile_struct.keys()
    ras, decs = [], []
    for key in keys:
        ras.append(tile_struct[key]["ra"])
        decs.append(tile_struct[key]["dec"])
    ras, decs = np.array(ras), np.array(decs)

    prob = np.zeros((len(keys),))
    for ii, key in enumerate(keys):
        prob[ii] = prob[ii] + tile_struct[key]['prob']

    sort_idx = np.argsort(prob)[::-1]
    csm = np.empty(len(prob))
    csm[sort_idx] = np.cumsum(prob[sort_idx])
    ipix_keep = np.where(csm <= params["iterativeOverlap"])[0]
 
    catalog1 = SkyCoord(ra=coverage_ras*u.degree,
                        dec=coverage_decs*u.degree, frame='icrs')

    for ii, key in enumerate(keys):
        # in the golden tile set
        if ii in ipix_keep: continue

        catalog2 = SkyCoord(ra=tile_struct[key]["ra"]*u.degree,
                            dec=tile_struct[key]["dec"]*u.degree,
                            frame='icrs')
        sep = catalog1.separation(catalog2)
        galaxies = tile_struct[key]["galaxies"]
        for jj, s in enumerate(sep):
            if s.deg > 1:
                continue
            galaxies2 = coverage_struct["galaxies"][jj]
            overlap = np.intersect1d(galaxies, galaxies2)
            if len(overlap) == 0:
                continue

            rat = np.array([float(len(overlap)) / float(len(galaxies)),
                            float(len(overlap)) / float(len(galaxies2))])

            if np.max(rat) > params["maximumOverlap"]:
                tile_struct[key]['prob'] = 0.0
                break

    return tile_struct

def optimize_max_tiles(params,opt_tile_struct,opt_coverage_struct,config_struct,telescope,map_struct_hold):
    """Returns value of max_tiles_nb optimized for number of scheduled fields with 'balanced' exposures in each filter."""
    prob={}
    for key in opt_tile_struct.keys():
        prob[key] = opt_tile_struct[key]['prob']
    
    tile_struct_hold = copy.copy(opt_tile_struct)
    keys_scheduled = opt_coverage_struct["data"][:,5]
    unique, freq = np.unique(keys_scheduled, return_counts=True)
    n_equal = np.sum(freq==len(params["filters"]))
    n_dif = np.sum(freq!=len(params["filters"]))

    optimized_max=-1 #assigns baseline optimized maxtiles
    n_dif_og = np.sum(freq!=len(params["filters"]))
    params["doMaxTiles"] = True
    countervals=[]
    
    coarse_bool = False
    repeating = False
    if (config_struct["FOV_type"] == "circle" and config_struct["FOV"] <= 2.0
        or config_struct["FOV_type"] == "square" and config_struct["FOV"] <= 4.0):
        max_trials = np.linspace(10,210,9)
        coarse_bool = True
    else:
        max_trials = np.linspace(10,200,20)

    for ii,max_trial in enumerate(max_trials):
        for key in tile_struct_hold.keys():
            tile_struct_hold[key]['prob'] = prob[key]
            if "epochs" in tile_struct_hold[key]:
                tile_struct_hold[key]['epochs'] = np.empty((0,9))
        params["max_nb_tiles"] = np.array([max_trial],dtype=np.float)
        params_hold = copy.copy(params)
        config_struct_hold = copy.copy(config_struct)
        coverage_struct_hold,tile_struct_hold = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold,
                                                                                  telescope, map_struct_hold, tile_struct_hold)

        keys_scheduled = coverage_struct_hold["data"][:,5]
        unique, freq = np.unique(keys_scheduled, return_counts=True)
        counter = np.sum(freq==len(params["filters"]))

        countervals.append(counter)

        #check for breaking conditions
        if counter >= n_equal:
            n_dif = np.sum(freq!=len(params["filters"]))
            if counter>n_equal or counter == n_equal and optimized_max == - 1 and n_dif <= n_dif_og:
                n_equal,optimized_max = counter,max_trial
                n_dif = np.sum(freq!=len(params["filters"]))
                opt_coverage_struct,opt_tile_struct = coverage_struct_hold,tile_struct_hold
        if ii>2:
            repeating = countervals[ii] == countervals[ii-1] == countervals[ii-2]

        if ii>0 and counter<countervals[ii-1] or repeating: break

    #optimize within narrower range for more precision
    if coarse_bool == True:
        max_trials = np.linspace(optimized_max,optimized_max+24,4)
    else:
        if optimized_max < 100:
            max_trials = np.linspace(optimized_max-3,optimized_max+9,7)
        elif optimized_max == 200:
            max_trials = np.linspace(optimized_max,optimized_max+60,4)
        else:
            max_trials = np.linspace(optimized_max,optimized_max+9,4)

    countervals=[]
    repeating = False
    for ii,max_trial in enumerate(max_trials):
        if optimized_max==-1: break #breaks if no max tiles restriction should be imposed
        for key in tile_struct_hold.keys():
            tile_struct_hold[key]['prob'] = prob[key]
            if "epochs" in tile_struct_hold[key]:
                tile_struct_hold[key]['epochs'] = np.empty((0,9))
        params["max_nb_tiles"] = np.array([max_trial],dtype=np.float)
        params_hold = copy.copy(params)
        config_struct_hold = copy.copy(config_struct)

        coverage_struct_hold,tile_struct_hold = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold,
                                                                                  telescope, map_struct_hold, tile_struct_hold)

        keys_scheduled = coverage_struct_hold["data"][:,5]
        unique, freq = np.unique(keys_scheduled, return_counts=True)
        counter = np.sum(freq==len(params["filters"]))

        countervals.append(counter)

        if counter>n_equal:
            n_equal,optimized_max = counter,max_trial
            n_dif = np.sum(freq!=len(params["filters"]))
            opt_coverage_struct,opt_tile_struct = coverage_struct_hold,tile_struct_hold
        if counter == n_equal and ii > 1:
            repeating = countervals[ii] == countervals[ii-1] == countervals[ii-2]

        if ii>0 and counter<countervals[ii-1] or repeating: break
            
    #check percent difference between # of fields scheduled in each filter
    n_1_og = n_equal
    n_2 = n_equal + n_dif
    p_dif = n_dif/((n_1_og+n_2)*0.5)

    if p_dif >=0.1:
        count = 0
        n_difs,n_equals,p_difs = [n_dif],[n_equal],[p_dif]
        unbalanced_tiles = []
        params_hold = copy.copy(params)
        
        while count<20:
            count+=1

            for key in tile_struct_hold.keys():
                tile_struct_hold[key]['prob'] = prob[key]
                if "epochs" in tile_struct_hold[key]:
                    tile_struct_hold[key]['epochs'] = np.empty((0,9))
            doReschedule,balanced_fields = balance_tiles(params_hold, opt_tile_struct, opt_coverage_struct)
            params_hold["unbalanced_tiles"] = unbalanced_tiles + params_hold["unbalanced_tiles"]
            if not doReschedule: break
            config_struct_hold = copy.copy(config_struct)
            params_hold["max_nb_tiles"] = np.array([np.ceil(optimized_max)],dtype=np.float)
            coverage_struct,tile_struct_hold = gwemopt.scheduler.schedule_alternating(params_hold, config_struct_hold, telescope,
                                                                                    map_struct_hold, tile_struct_hold)
            keys_scheduled = coverage_struct["data"][:,5]
            unique, freq = np.unique(keys_scheduled, return_counts=True)
            n_equal = np.sum(freq==len(params["filters"]))
            n_dif = np.sum(freq!=len(params["filters"]))
            n_1 = n_equal
            n_2 = n_equal + n_dif
            p_dif = n_dif/((n_1+n_2)*0.5)

            if p_dif > p_difs[-1] and p_dif >= 0.15 and optimized_max > 0: #try decreasing max-tiles if n_difs increase
                optimized_max -= 0.1*optimized_max
                continue
            elif (p_dif > p_difs[-1]) or (p_difs[-1] < 0.15 and n_equal < n_equals[-1]): break
            opt_coverage_struct,opt_tile_struct = coverage_struct,tile_struct_hold
            n_difs.append(n_dif)
            n_equals.append(n_equal)
            p_difs.append(p_dif)
            unbalanced_tiles = unbalanced_tiles + params_hold["unbalanced_tiles"]

            if count == 19 and np.min(p_difs)>0.15: #try setting it to original n_equal as final resort
                optimized_max = n_1_og

    for key in tile_struct_hold.keys():
        tile_struct_hold[key]['prob'] = prob[key]

    return optimized_max,opt_coverage_struct,opt_tile_struct

def check_overlapping_tiles(params, tile_struct, coverage_struct):

    coverage_ras = coverage_struct["data"][:,0]
    coverage_decs = coverage_struct["data"][:,1]
#   coverage_mjds = coverage_struct["data"][:,2]
    coverage_ipixs = coverage_struct["ipix"]
    if len(coverage_ras) == 0:
        return tile_struct

    keys = list(tile_struct.keys())
    ras, decs = [], []
    for key in keys:
        ras.append(tile_struct[key]["ra"])
        decs.append(tile_struct[key]["dec"])
    ras, decs = np.array(ras), np.array(decs)

    catalog1 = SkyCoord(ra=coverage_ras*u.degree,
                        dec=coverage_decs*u.degree, frame='icrs')
    if params["tilesType"] == "galaxy":
        for ii, key in enumerate(keys):
            catalog2 = SkyCoord(ra=tile_struct[key]["ra"]*u.degree,
                                dec=tile_struct[key]["dec"]*u.degree,
                                frame='icrs')
            sep = catalog1.separation(catalog2)
            galaxies = tile_struct[key]["galaxies"]
            for jj, s in enumerate(sep):
                if s.deg > 1:
                    continue
                galaxies2 = coverage_struct["galaxies"][jj]
                overlap = np.setdiff1d(galaxies, galaxies2)
                if len(overlap) == 0:
                    if not 'epochs' in tile_struct[key]:
                        tile_struct[key]["epochs"] = np.empty((0,9))
                    tile_struct[key]["epochs"] = np.append(tile_struct[key]["epochs"],np.atleast_2d(coverage_struct["data"][jj,:]),axis=0)
    else:
        for ii, key in enumerate(keys):
            catalog2 = SkyCoord(ra=tile_struct[key]["ra"]*u.degree,
                                dec=tile_struct[key]["dec"]*u.degree,
                                frame='icrs')
            sep = catalog1.separation(catalog2)
            ipix = tile_struct[key]["ipix"]
            for jj, s in enumerate(sep):
                if s.deg > 25:
                    continue
                ipix2 = coverage_struct["ipix"][jj]
                overlap = np.intersect1d(ipix, ipix2)
                
                rat = np.array([float(len(overlap)) / float(len(ipix)),
                                float(len(overlap)) / float(len(ipix2))])
                                
                if len(overlap) == 0 or (params["doSuperSched"] and np.max(rat) < 0.50): continue
                if params["doSuperSched"]:
                    if 'epochs_telescope' not in tile_struct[key]:
                        tile_struct[key]["epochs_telescope"]=[]
                    tile_struct[key]["epochs_telescope"].append(coverage_struct["telescope"][jj])

                if not 'epochs' in tile_struct[key]:
                    tile_struct[key]["epochs"] = np.empty((0,9))
                    tile_struct[key]["epochs_overlap"] = []
                    tile_struct[key]["epochs_filters"]=[]
                    
                tile_struct[key]["epochs"] = np.append(tile_struct[key]["epochs"],np.atleast_2d(coverage_struct["data"][jj,:]),axis=0)
                tile_struct[key]["epochs_overlap"].append(len(overlap))
                tile_struct[key]["epochs_filters"].append(coverage_struct["filters"][jj])

    return tile_struct

def append_tile_epochs(tile_struct,coverage_struct):
    for key in tile_struct.keys():
        if key not in coverage_struct["data"][:,5]: continue
        if not 'epochs' in tile_struct[key]:
            tile_struct[key]["epochs"] = np.empty((0,9))
        idx = np.where(coverage_struct["data"][:,5] == key)[0]
        for jj in idx:
            tile_struct[key]["epochs"] = np.append(tile_struct[key]["epochs"],np.atleast_2d(coverage_struct["data"][jj,:]),axis=0)

    return tile_struct

def order_by_observability(params,tile_structs):
    observability = []
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tiles_struct = tile_structs[telescope]
        exposurelist = config_struct["exposurelist"]
        observability_prob = 0.0
        keys = tiles_struct.keys()
        for jj, key in enumerate(keys):
            tilesegmentlist = tiles_struct[key]["segmentlist"]
            if tiles_struct[key]["prob"] == 0: continue
            if tilesegmentlist.intersects_segment(exposurelist[0]):
                observability_prob = observability_prob + tiles_struct[key]["prob"]
        observability.append(observability_prob)
    idx = np.argsort(observability)[::-1]
    params["telescopes"] = [params["telescopes"][ii] for ii in idx]

def get_treasuremap_pointings(params):
    BASE = 'http://treasuremap.space/api/v0'
    TARGET = 'pointings'
    info = {
        "api_token": params["treasuremap_token"],
        "bands": params["filters"],
        "statuses": params["treasuremap_status"],
        "graceid": params["graceid"]
    }
    
    url = "{}/{}?{}".format(BASE, TARGET, urllib.parse.urlencode(info))

    try:
        r = requests.get(url = url)
    except requests.exceptions.RequestException as e:
        print(e)
        exit(1)

    observations = r.text.split("}")

    #dicts of instrument FOVs
    FOV_square = {44: 4.96, 47: 6.86}
    FOV_circle = {38: 1.1}
    
    #create coverage_struct
    coverage_struct = {}
    coverage_struct["data"] = np.empty((0, 8))
    coverage_struct["filters"] = []
    coverage_struct["ipix"] = []

    #read information into coverage struct
    for obs in observations:
        if "invalid api_token" in obs:
            print("Invalid Treasure Map API token.")
            exit(1)
        elif "POINT" not in obs: continue
        
        pointing = re.search('\(([^)]+)', obs).group(1)
        pointing = pointing.split(" ")
        ra, dec = float(pointing[0]), float(pointing[1])
        
        filteridx = obs.find("band") + 10 #jump to starting index of filter
        filter = obs[filteridx:].split("\"")[0][:-1]

        instrumentidx = obs.find("instrumentid") + 16 #jump to starting index of instrument id
        instrument_id = int(obs[instrumentidx:].split(",")[0])

        if instrument_id in FOV_square:
            ipix, radecs, patch, area = gwemopt.utils.getSquarePixels(ra, dec, FOV_square[instrument_id], params["nside"])
        elif instrument_id in FOV_circle:
            ipix, radecs, patch, area = gwemopt.utils.getCirclePixels(ra, dec, FOV_circle[instrument_id], params["nside"])
        else: continue
        
        coverage_struct["data"] = np.append(coverage_struct["data"], np.array([[ra, dec, -1, -1, -1, -1, -1, -1]]), axis=0)
        coverage_struct["filters"].append(filter)
        coverage_struct["ipix"].append(ipix)

    return coverage_struct

