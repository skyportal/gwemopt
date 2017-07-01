
import os, sys
import numpy as np
import healpy as hp

from scipy.stats import norm

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.path

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
                line_split = filter(None, line_split)
                if line_split:
                    try:
                        params[line_split[0]] = float(line_split[1])
                    except:
                        params[line_split[0]] = line_split[1]
    return params

def read_skymap(params,filename,is3D=False):

    map_struct = {}

    if is3D:
        healpix_data = hp.read_map(filename, field=(0,1,2,3))

        distmu_data = healpix_data[1]
        diststd_data = healpix_data[2]
        prob_data = healpix_data[0]
        norm_data = healpix_data[3]

        map_struct["distmu"] = distmu_data / params["DScale"]
        map_struct["diststd"] = diststd_data / params["DScale"]
        map_struct["prob"] = prob_data
        map_struct["distnorm"] = norm_data


    else:
        prob_data = hp.read_map(filename, field=0)
        prob_data = prob_data / np.sum(prob_data)

        map_struct["prob"] = prob_data

    nside = hp.pixelfunc.get_nside(prob_data)
    nside = params["nside"]
    map_struct["prob"] = hp.ud_grade(map_struct["prob"],nside)
    if is3D:
        map_struct["distmu"] = hp.ud_grade(map_struct["distmu"],nside) 
        map_struct["diststd"] = hp.ud_grade(map_struct["diststd"],nside) 
        map_struct["distnorm"] = hp.ud_grade(map_struct["distnorm"],nside) 

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5*np.pi - theta)

    map_struct["ra"] = ra
    map_struct["dec"] = dec

    sort_idx = np.argsort(prob_data)[::-1]
    csm = np.empty(len(prob_data))
    csm[sort_idx] = np.cumsum(prob_data[sort_idx])

    map_struct["cumprob"] = csm

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

    for ii in xrange(Nsamples):
        ipix = np.argmin(np.abs(prob_data_cumsum-rand_values[ii]))
        ra_inj = map_struct["ra"][prob_data_indexes][ipix]
        dec_inj = map_struct["dec"][prob_data_indexes][ipix]

        ras.append(ra_inj)
        decs.append(dec_inj)    

        if is3D:
            dp_dr = r**2 * map_struct["distnorm"][prob_data_indexes][ipix] * norm(map_struct["distmu"][prob_data_indexes][ipix], map_struct["diststd"][prob_data_indexes][ipix]).pdf(r)
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

def getCirclePixels(ra_pointing, dec_pointing, radius, nside):

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

    xyz = hp.ang2vec(radecs[:,0],radecs[:,1],lonlat=True)

    proj = hp.projector.MollweideProj(rot=None, coord=None)
    x,y = proj.vec2xy(xyz[:,0],xyz[:,1],xyz[:,2])
    xy = np.zeros(radecs.shape)
    xy[:,0] = x
    xy[:,1] = y
    #path = matplotlib.path.Path(xyz[:,1:3])
    path = matplotlib.path.Path(xy)
    patch = matplotlib.patches.PathPatch(path, alpha=0.2, color='#6c71c4', fill=True, zorder=3,)

    return ipix, radecs, patch

def getSquarePixels(ra_pointing, dec_pointing, tileSide, nside):

    decCorners = (dec_pointing - tileSide / 2.0, dec_pointing + tileSide / 2.0)
 
    radecs = []
    xyz = []
    for d in decCorners:
        if d > 90.:
            d = 180. - d
        elif d < -90.:
            d = -180 - d

        raCorners = (ra_pointing - (tileSide / 2.0) / np.cos(np.deg2rad(d)) , ra_pointing + (tileSide / 2.0) / np.cos(np.deg2rad(d)))

        for r in raCorners:
            if r > 360.:
                r = 720. - r
            elif r < 0.:
                r = 360. + r
            xyz.append(hp.ang2vec(r, d, lonlat=True))

            radecs.append([r,d])
    radecs = np.array(radecs)

    # FLIP CORNERS 3 & 4 SO HEALPY UNDERSTANDS POLYGON SHAPE
    xyz = [xyz[0], xyz[1],
           xyz[3], xyz[2]]

    # RETURN HEALPIXELS IN EXPOSURE AREA
    ipix = hp.query_polygon(nside, np.array(
        xyz))

    xyz = np.array(xyz)
    proj = hp.projector.MollweideProj(rot=None, coord=None) 
    x,y = proj.vec2xy(xyz[:,0],xyz[:,1],xyz[:,2])
    xy = np.zeros(radecs.shape)
    xy[:,0] = x
    xy[:,1] = y
    path = matplotlib.path.Path(xy)
    patch = matplotlib.patches.PathPatch(path, alpha=0.2, color='#6c71c4', fill=True, zorder=3,)
    
    return ipix, radecs, patch
