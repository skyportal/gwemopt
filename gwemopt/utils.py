
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

import astropy.coordinates
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.coordinates import SkyCoord

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

import gwemopt.moc
import gwemopt.tiles
import gwemopt.segments

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
    #assigns defaults to params
    do_Parameters = ["do3D","doEvent","doSuperSched","doMovie_supersched","doSkymap","doSamples","doCoverage","doSchedule","doPlots","doDatabase","doMovie","doTiles","doIterativeTiling","doMinimalTiling","doOverlappingScheduling","doPerturbativeTiling","doOrderByObservability","doCatalog","doUseCatalog","doCatalogDatabase","doObservability","doSkybrightness","doEfficiency","doCalcTiles","doTransients","doSingleExposure","doAlternatingFilters","doMaxTiles","doReferences","doChipGaps","doUsePrimary","doSplit","doParallel","writeCatalog","doFootprint","doBalanceExposure"]
 
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

    if "powerlaw_c1" not in params.keys():
        params["powerlaw_c1"] = 0.9

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
        params["transientsFile"] = "../transients/ps1_objects.csv"

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

    if "splitType" not in params.keys():
        params["splitType"] = "regional"

    if "Nregions" not in params.keys():
        params["Nregions"] = 768

    if "configDirectory" not in params.keys():
        params["configDirectory"] = "../config/"

    if "Ncores" not in params.keys():
        params["Ncores"] = 4

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
            
                nmag = np.log(exposuretime/params["config"][telescope]["exposuretime"]) / np.log(2.5)
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

def read_skymap(params,is3D=False,map_struct=None):

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
                healpix_data = hp.read_map(filename, field=(0,1,2,3), verbose=False)
        
                distmu_data = healpix_data[1]
                distsigma_data = healpix_data[2]
                prob_data = healpix_data[0]
                norm_data = healpix_data[3]
        
                map_struct["distmu"] = distmu_data / params["DScale"]
                map_struct["distsigma"] = distsigma_data / params["DScale"]
                map_struct["prob"] = prob_data
                map_struct["distnorm"] = norm_data
    
            else:
                prob_data = hp.read_map(filename, field=0, verbose=False)
                prob_data = prob_data / np.sum(prob_data)
        
                map_struct["prob"] = prob_data

    natural_nside = hp.pixelfunc.get_nside(map_struct["prob"])
    nside = params["nside"]
    
    print("natural_nside =", natural_nside)
    print("nside =", nside)
    
    if not is3D:
        map_struct["prob"] = hp.ud_grade(map_struct["prob"],nside,power=-2)

    if is3D:
        if natural_nside != nside:
            map_struct["prob"], map_struct["distmu"],\
            map_struct["distsigma"], map_struct["distnorm"] = ligodist.ud_grade(map_struct["prob"],\
                                                                                map_struct["distmu"],\
                                                                                map_struct["distsigma"],\
                                                                                nside)
        
        nside_down = 32
        _, distmu_down,\
        distsigma_down, distnorm_down = ligodist.ud_grade(map_struct["prob"],\
                                                          map_struct["distmu"],\
                                                          map_struct["distsigma"],\
                                                          nside_down)

        r = np.linspace(0, 2000)
        map_struct["distmed"] = np.zeros(distmu_down.shape)
        for ipix in range(len(map_struct["distmed"])):
            dp_dr = r**2 * distnorm_down[ipix] * norm(distmu_down[ipix],distsigma_down[ipix]).pdf(r)
            dp_dr_norm = np.cumsum(dp_dr / np.sum(dp_dr))
            idx = np.argmin(np.abs(dp_dr_norm-0.5))
            map_struct["distmed"][ipix] = r[idx]
        map_struct["distmed"] = hp.ud_grade(map_struct["distmu"],nside,power=-2)

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5*np.pi - theta)

    map_struct["ra"] = ra
    map_struct["dec"] = dec

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
            idx = np.setdiff1d(idx,map_struct_hold["ipix_keep"])
            if len(map_struct_hold["prob"][moc_struct_temp["ipix"]]) == 0:
                rat = 0.0
            else:
                rat = float(len(idx)) / float(len(map_struct_hold["prob"][moc_struct_temp["ipix"]]))
            if rat > params["maximumOverlap"]:
                val = 0.0
            else:
                vals_to_sum = map_struct_hold["prob"][moc_struct_temp["ipix"]]
                vals_to_sum[vals_to_sum < 0] = 0
                val = np.sum(vals_to_sum) 
            vals.append(val)
        idx = np.argmax(vals)
        ra, dec = ras[idx], decs[idx]
        moc_struct[key] = gwemopt.moc.Fov2Moc(params, config_struct, telescope, ra, dec, nside)

        map_struct_hold['prob'][moc_struct[key]["ipix"]] = -1

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

def balance_tiles(params, telescope, tile_struct, coverage_struct):

    filters, exposuretimes = params["filters"], params["exposuretimes"]

    keys_scheduled = coverage_struct["data"][:,5]
    filts = coverage_struct["filters"]
    filts_used = {key: [] for key in keys_scheduled} #dictionary of scheduled filters for each key

    for (key,filt) in zip(keys_scheduled,filts):
        filts_used[key].append(filt)

    doReschedule = False
    for key in filts_used:
        if len(filts_used[key]) != len(filters):
            tile_struct[key]['prob'] = 0.0
            doReschedule = True

    return tile_struct, doReschedule

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

def check_overlapping_tiles(params, tile_struct, coverage_struct):

    coverage_ras = coverage_struct["data"][:,0]
    coverage_decs = coverage_struct["data"][:,1]
    coverage_mjds = coverage_struct["data"][:,2]
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
                        tile_struct[key]["epochs"] = np.empty((0,8))
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
                                
                if params["doSuperSched"]:
                    if np.max(rat) < 0.99:
                        continue
                elif len(overlap)==0:
                        continue

                if not 'epochs' in tile_struct[key]:
                    tile_struct[key]["epochs"] = np.empty((0,8))
                    tile_struct[key]["epochs_telescope"] = np.empty((0,1))
                tile_struct[key]["epochs"] = np.append(tile_struct[key]["epochs"],np.atleast_2d(coverage_struct["data"][jj,:]),axis=0)
                tile_struct[key]["epochs_telescope"] = np.append(tile_struct[key]["epochs_telescope"],np.atleast_2d(coverage_struct["telescope"][jj]),axis = 0)

    return tile_struct
