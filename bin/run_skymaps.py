# -*- coding: utf-8 -*-
# Copyright (C) Michael Coughlin and Christopher Stubbs(2015)
#
# skybrightness is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gwemopt is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with skybrightness.  If not, see <http://www.gnu.org/licenses/>.

"""This module provides example methods to calculate the optimal likelihood and distance scalings for optimizing telescope time allocations.
"""

from __future__ import division

import os, sys, pickle, math, optparse, glob, time, subprocess
from datetime import date, datetime
import numpy as np

seed = 1
np.random.seed(seed=seed)

import scipy.ndimage
import astropy

import healpy as hp

import scipy.io
import h5py

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.pyplot as plt
from matplotlib import cm as cmx
from matplotlib import colors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.mlab import griddata

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pymultinest

__author__ = "Michael Coughlin <michael.coughlin@ligo.org>"
__version__ = 1.0
__date__    = "9/22/2013"

def parse_commandline():
    """@Parse the options given on the command-line.
    """
    parser = optparse.OptionParser(usage=__doc__,version=__version__)

    parser.add_option("-o","--outputDir",help="Output location",default = '/home/mcoughlin/Skymaps/optimization/plots')
    parser.add_option("-d","--dataDir",help="Data location",default  = '/home/mcoughlin/Skymaps/going-the-distance_data/2015/compare/')

    parser.add_option("-t", "--telescope", help="Telescope.",
                      default ="PS1")

    parser.add_option("--minabsm", help="Min absolute magnitude.",default=-16,type=float)
    parser.add_option("--maxabsm", help="Max absolute magnitude.",default=-10,type=float)    
    parser.add_option("--minfade", help="Min time above detection threshold [hours].",default=1,type=float)
    parser.add_option("--maxfade", help="Max time above detection threshold [hours].",default=10,type=float)

    parser.add_option("-e", "--event", help="Event.",default = None)
    parser.add_option("-n", "--nsignals", help="Number of simulated signals.", type=int, default = 10000)

    parser.add_option("-v", "--verbose", action="store_true", default=False,
                      help="Run verbosely. (Default: False)")

    opts, args = parser.parse_args()

    # show parameters
    if opts.verbose:
        print >> sys.stderr, ""
        print >> sys.stderr, "running pylal_seismon_run..."
        print >> sys.stderr, "version: %s"%__version__
        print >> sys.stderr, ""
        print >> sys.stderr, "***************** PARAMETERS ********************"
        for o in opts.__dict__.items():
          print >> sys.stderr, o[0]+":"
          print >> sys.stderr, o[1]
        print >> sys.stderr, ""

    return opts


def get_post_file(basedir):
    filenames = glob.glob(os.path.join(basedir,'2-post*'))
    if len(filenames)>0:
        filename = filenames[0]
    else:
        filename = []
    return filename

def myprior(cube, ndim, nparams):
        cube[0] = cube[0]*8.0 - 4.0
        cube[1] = cube[1]*1.0
        cube[2] = cube[2]*8.0 - 4.0

def myprior_multi(cube, ndim, nparams):
        cube[0] = cube[0]*8.0 - 4.0
        cube[1] = cube[1]*1.0
        cube[2] = cube[2]*8.0 - 4.0
        cube[3] = cube[3]*8.0 - 4.0
        cube[4] = cube[4]*1.0
        cube[5] = cube[5]*8.0 - 4.0

def myloglike(cube, ndim, nparams):
        n = cube[0]
        cl = cube[1]
        r = cube[2]

        image_array = numimages(prob_data,distmu_data,data_out,Telescope_T,Telescope_m,n,cl,r)

        image_array_sorted = np.sort(image_array)

        tmax = data_out["fade"]*3600
        image_detected = float(len(np.where(image_array < tmax)[0]))/float(len(image_array))

        index = np.floor(len(image_array)*0.5)
        image_T = image_array_sorted[index]/3600.0

        #prob = 1/image_detected
        prob = image_detected
        #prob = -np.log10(1-image_detected)
        #prob = -np.log10(image_T)

        #if np.isfinite(prob):
        #    print n, cl, r, prob
        return prob

def myloglike_multi(cube, ndim, nparams):
        n_PS1 = cube[0]
        cl_PS1 = cube[1]
        r_PS1 = cube[2]
        n_ATLAS = cube[3]
        cl_ATLAS = cube[4]
        r_ATLAS = cube[5]

        image_array = numimages_combine(prob_data,distmu_data,data_out,n_PS1,cl_PS1,r_PS1,n_ATLAS,cl_ATLAS,r_ATLAS)

        image_array_sorted = np.sort(image_array)

        tmax = data_out["fade"]*3600
        image_detected = float(len(np.where(image_array < tmax)[0]))/float(len(image_array))

        index = np.floor(len(image_array)*0.5)
        image_T = image_array_sorted[index]/3600.0

        prob = image_detected

        #if np.isfinite(prob):
        #    print n_PS1, cl_PS1, r_PS1, n_ATLAS, cl_ATLAS, r_ATLAS, prob
        return prob

def probpower(prob_data,distmu_data,n,cl,r):

    prob_data_sorted = np.sort(prob_data)[::-1]
    prob_data_indexes = np.argsort(prob_data)[::-1]
    prob_data_cumsum = np.cumsum(prob_data_sorted)
    index = np.argmin(np.abs(prob_data_cumsum - cl)) + 1

    prob_data_scaled = prob_data.copy()
    prob_data_scaled[prob_data_indexes[index:]] = 0.0
    prob_data_scaled = prob_data_scaled**n
    distmu_data_scaled = (distmu_data/np.nanmax(distmu_data))**r
    prob_data_scaled = prob_data_scaled * distmu_data_scaled
    prob_data_scaled[np.isnan(prob_data_scaled)] = 0.0
    prob_data_scaled = prob_data_scaled / np.nansum(prob_data_scaled)

    return prob_data_scaled 

def numimages(prob_data,distmu_data,data_out,Telescope_T,Telescope_m,n,cl,r):

    phi_true = data_out["ra"]
    theta_true = 0.5*np.pi - data_out["dec"]
    index_true = hp.ang2pix(nside, theta_true, phi_true)
    appm = data_out["absm"] + 5*(np.log10(data_out["dist"]*1e6) - 1)

    Telescope_nimages = 2.5**(appm - Telescope_m)
    Telescope_T_scaled = Telescope_T*Telescope_nimages

    prob_data_scaled = probpower(prob_data,distmu_data,n,cl,r)
    prob_data_true = prob_data_scaled[index_true]

    image_array = Telescope_T_scaled/prob_data_true

    return image_array

def numimages_combine(prob_data,distmu_data,data_out,n_PS1,cl_PS1,r_PS1,n_ATLAS,cl_ATLAS,r_ATLAS):
    image_array_PS1 = numimages(prob_data,distmu_data,data_out,PS1_Telescope_T,PS1_Telescope_m,n_PS1,cl_PS1,r_PS1)
    image_array_ATLAS = numimages(prob_data,distmu_data,data_out,ATLAS_Telescope_T,ATLAS_Telescope_m,n_ATLAS,cl_ATLAS,r_ATLAS)
    image_array = np.min(np.vstack((image_array_PS1,image_array_ATLAS)),axis=0)

    return image_array

def numknown(prob_data,data_out,Telescope_T,Telescope_m):

    phi_true = data_out["ra"]
    theta_true = 0.5*np.pi - data_out["dec"]
    index_true = hp.ang2pix(nside, theta_true, phi_true)
    appm = data_out["absm"] + 5*(np.log10(data_out["dist"]*1e6) - 1)

    Telescope_nimages = 2.5**(appm - Telescope_m)
    Telescope_T_scaled = Telescope_T*Telescope_nimages

    return Telescope_T_scaled

def hist_results(samples):

    bins = np.linspace(np.min(samples),np.max(samples),11)
    hist1, bin_edges = np.histogram(samples, bins=bins)
    hist1 = hist1 / float(np.sum(hist1))
    bins = (bins[1:] + bins[:-1])/2.0

    return bins, hist1

def healpix(filename,plotDir,data_out):

    bayestar = True
    if bayestar:
        healpix_data = hp.read_map(filename, field=(0,1,2,3))

        distmu_data = healpix_data[1]
        diststd_data = healpix_data[2]
        prob_data = healpix_data[0]
        norm_data = healpix_data[3]

        plotName = os.path.join(plotDir,'mollview.png')
        vmin = np.min(prob_data)
        vmax = np.max(prob_data)
        hp.mollview(prob_data, min=vmin,max=vmax,title="",unit='Likelihood')
        hp.graticule()
        plt.show()
        plt.savefig(plotName,dpi=200)
        plotName = os.path.join(plotDir,'mollview.eps')
        plt.savefig(plotName,dpi=200)
        plotName = os.path.join(plotDir,'mollview.pdf')
        plt.savefig(plotName,dpi=200)
        plt.close('all')

        distmu_data[distmu_data<0] = np.inf
        distmu_data_cut = distmu_data[~np.isinf(distmu_data)] 

        plotName = os.path.join(plotDir,'mollview_dist.png')
        vmin = np.min(distmu_data_cut)
        vmax = np.max(distmu_data_cut)

        hp.mollview(distmu_data, min=vmin,max=vmax,title="",unit='Distance [Mpc]')
        hp.graticule()
        plt.show()
        plt.savefig(plotName,dpi=200)
        plotName = os.path.join(plotDir,'mollview_dist.eps')
        plt.savefig(plotName,dpi=200)
        plotName = os.path.join(plotDir,'mollview_dist.pdf')
        plt.savefig(plotName,dpi=200)
        plt.close('all')

        diststd_data[diststd_data<0] = np.inf
        diststd_data_cut = diststd_data[~np.isinf(diststd_data)]

        plotName = os.path.join(plotDir,'mollview_dist_std.png')
        vmin = np.min(diststd_data_cut)
        vmax = np.max(diststd_data_cut)

        hp.mollview(diststd_data, min=vmin,max=vmax,title="",unit='Distance [Mpc]')
        hp.graticule()
        plt.show()
        plt.savefig(plotName,dpi=200)
        plotName = os.path.join(plotDir,'mollview_dist_std.eps')
        plt.savefig(plotName,dpi=200)
        plotName = os.path.join(plotDir,'mollview_dist_std.pdf')
        plt.savefig(plotName,dpi=200)
        plt.close('all')

        distmu_data = hp.ud_grade(distmu_data,nside)
        diststd_data = hp.ud_grade(diststd_data,nside)
        norm_data = hp.ud_grade(norm_data,nside)
    else:
        prob_data = hp.read_map(filename, field=0)
        prob_data = prob_data / np.sum(prob_data)    
    prob_data = hp.ud_grade(prob_data,nside)
    prob_data = prob_data / np.sum(prob_data)

    indexes = np.where(prob_data < np.max(prob_data)*0.01)[0]
    prob_data_nan = prob_data.copy()
    prob_data_nan[indexes] = np.nan
    vmin = np.nanmin(prob_data_nan)
    vmax = np.nanmax(prob_data_nan)

    plotName = os.path.join(plotDir,'mollview_small.png')
    hp.mollview(prob_data_nan, min=vmin,max=vmax,title="",unit='Likelihood')
    hp.graticule()
    plt.show()
    plt.savefig(plotName,dpi=200)
    plotName = os.path.join(plotDir,'mollview_small.eps')
    plt.savefig(plotName,dpi=200)
    plotName = os.path.join(plotDir,'mollview_small.pdf')
    plt.savefig(plotName,dpi=200)
    plt.close('all')
   
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5*np.pi - theta)
    absm = np.random.rand(len(data_out["ra"]),)*(maxabsm-minabsm) + minabsm
    data_out["absm"] = absm
    fade = np.random.rand(len(data_out["ra"]),)*(maxfade-minfade) + minfade
    data_out["fade"] = fade

    # Run MC
    print "Ra: %.5f %.5f"%(np.min(data_out["ra"]),np.max(data_out["ra"]))
    print "Declination: %.5f %.5f"%(np.min(data_out["dec"]),np.max(data_out["dec"]))
    print "Distance: %.5f %.5f"%(np.min(data_out["dist"]),np.max(data_out["dist"]))
    print "Absolute Magnitude: %.5f %.5f"%(np.min(data_out["absm"]),np.max(data_out["absm"]))  

    if telescope == "combined":
        # number of dimensions our problem has
        parameters = ["n_PS1","cl_PS1","r_PS1","n_ATLAS","cl_ATLAS","r_ATLAS"]
    else:
        # number of dimensions our problem has
        parameters = ["n","cl","r"]
    n_params = len(parameters)

    global prob_data, distmu_data

    if telescope == "combined":
        pymultinest.run(myloglike_multi, myprior_multi, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = 1000, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = 0.001, multimodal = False)
    else:
        pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = 1000, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = 0.001, multimodal = False)

    # lets analyse the results
    a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='%s/2-'%plotDir)
    s = a.get_stats()

    import json
    # store name of parameters, always useful
    with open('%sparams.json' % a.outputfiles_basename, 'w') as f:
                json.dump(parameters, f, indent=2)
    # store derived stats
    with open('%sstats.json' % a.outputfiles_basename, mode='w') as f:
                json.dump(s, f, indent=2)
    print()
    print("-" * 30, 'ANALYSIS', "-" * 30)
    print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))

    #multifile= os.path.join(plotDir,'2-.txt')
    multifile = get_post_file(plotDir)
    data = np.loadtxt(multifile)

    #loglikelihood = -(1/2.0)*data[:,1]
    #idx = np.argmax(loglikelihood)

    if telescope == "combined":

        n_PS1 = data[:,0]
        cl_PS1 = data[:,1]
        r_PS1 = data[:,2]
        n_ATLAS = data[:,3]
        cl_ATLAS = data[:,4]
        r_ATLAS = data[:,5]
        loglikelihood = data[:,6]
        idx = np.argmax(loglikelihood)
    
        n_PS1_best = data[idx,0]
        cl_PS1_best = data[idx,1]
        r_PS1_best = data[idx,2]
        n_ATLAS_best = data[idx,3]
        cl_ATLAS_best = data[idx,4]
        r_ATLAS_best = data[idx,5]
    
        image_optimal_array = numimages_combine(prob_data,distmu_data,data_out,n_PS1_best,cl_PS1_best,r_PS1_best,n_ATLAS_best,cl_ATLAS_best,r_ATLAS_best)
        image_optimal_array_sorted = np.sort(image_optimal_array)
    
        image_nominal_array = numimages_combine(prob_data,distmu_data,data_out,0.0,0.99,0.0,0.0,0.99,0.0)
        image_nominal_array_sorted = np.sort(image_nominal_array)
    
        image_known_array = numknown(prob_data,data_out,PS1_Telescope_T,PS1_Telescope_m)
        image_known_array_sorted = np.sort(image_known_array)
    
        filename = os.path.join(plotDir,'samples.dat')
        fid = open(filename,'w+')
        for i, j, k, l, m, o in zip(n_PS1,cl_PS1,r_PS1,n_ATLAS,cl_ATLAS,r_ATLAS):
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,o))
        fid.close()
    
        filename = os.path.join(plotDir,'best.dat')
        fid = open(filename,'w')
        fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(n_PS1_best,cl_PS1_best,r_PS1_best,n_ATLAS_best,cl_ATLAS_best,r_ATLAS_best))
        fid.close()
    
        plt.figure(figsize=(12,10))
        bins1, hist1 = hist_results(n_PS1)
        bins2, hist2 = hist_results(n_ATLAS)
        #plt.plot(bins1, np.cumsum(hist1),'k',label='PS1')
        #plt.plot(bins2, np.cumsum(hist2),'k--',label='ATLAS')
        plt.plot(bins1, hist1,'k',label='PS1')
        plt.plot(bins2, hist2,'k--',label='ATLAS')
        plt.legend(loc='best')
        #plt.xlim([10,1e6])
        #plt.ylim([0,1.0])
        plt.xlabel('Likelihood Powerlaw Index')
        plt.ylabel('Probability Density Function')
        plt.show()
        plotName = os.path.join(plotDir,'n.pdf')
        plt.savefig(plotName,dpi=200)
        plt.close('all')
    
        plt.figure(figsize=(12,10))
        bins1, hist1 = hist_results(cl_PS1)
        bins2, hist2 = hist_results(cl_ATLAS)
        #plt.plot(bins1, np.cumsum(hist1),'k',label='PS1')
        #plt.plot(bins2, np.cumsum(hist2),'k--',label='ATLAS')
        plt.plot(bins1, hist1,'k',label='PS1')
        plt.plot(bins2, hist2,'k--',label='ATLAS')
        plt.legend(loc='best')
        #plt.xlim([10,1e6])
        #plt.ylim([0,1.0])
        plt.xlabel('Likelihood Confidence Level')
        plt.ylabel('Probability Density Function')
        plt.show()
        plotName = os.path.join(plotDir,'cl.pdf')
        plt.savefig(plotName,dpi=200)
        plt.close('all')
    
        plt.figure(figsize=(12,10))
        bins1, hist1 = hist_results(r_PS1)
        bins2, hist2 = hist_results(r_ATLAS)
        #plt.plot(bins1, np.cumsum(hist1),'k',label='PS1')
        #plt.plot(bins2, np.cumsum(hist2),'k--',label='ATLAS')
        plt.plot(bins1, hist1,'k',label='PS1')
        plt.plot(bins2, hist2,'k--',label='ATLAS')
        plt.legend(loc='best')
        #plt.xlim([10,1e6])
        #plt.ylim([0,1.0])
        plt.xlabel('Distance Powerlaw Index')
        plt.ylabel('Probability Density Function')
        plt.show()
        plotName = os.path.join(plotDir,'r.pdf')
        plt.savefig(plotName,dpi=200)
        plt.close('all')

    else:
        n = data[:,0]
        cl = data[:,1]
        r = data[:,2]
        loglikelihood = data[:,3]
        idx = np.argmax(loglikelihood)

        n_best = data[idx,0]
        cl_best = data[idx,1]
        r_best = data[idx,2]

        image_optimal_array = numimages(prob_data,distmu_data,data_out,Telescope_T,Telescope_m,n_best,cl_best,r_best)
        image_optimal_array_sorted = np.sort(image_optimal_array)

        image_nominal_array = numimages(prob_data,distmu_data,data_out,Telescope_T,Telescope_m,0.0,0.99,0.0)
        image_nominal_array_sorted = np.sort(image_nominal_array)

        image_known_array = numknown(prob_data,data_out,Telescope_T,Telescope_m)
        image_known_array_sorted = np.sort(image_known_array)

        filename = os.path.join(plotDir,'samples.dat')
        fid = open(filename,'w+')
        for i, j, k in zip(n,cl,r):
            fid.write('%.5f %.5f %.5f\n'%(i,j,k))
        fid.close()

        filename = os.path.join(plotDir,'best.dat')
        fid = open(filename,'w')
        fid.write('%.5f %.5f %.5f\n'%(n_best,cl_best,r_best))
        fid.close()

        plt.figure(figsize=(12,10))
        bins1, hist1 = hist_results(n)
        plt.plot(bins1, hist1)
        plt.xlabel('Powerlaw Index')
        plt.ylabel('Probability Density Function')
        plt.show()
        plotName = os.path.join(plotDir,'n.pdf')
        plt.savefig(plotName,dpi=200)
        plt.close('all')
    
        plt.figure(figsize=(12,10))
        bins1, hist1 = hist_results(cl)
        plt.plot(bins1, hist1)
        plt.xlabel('Powerlaw Index')
        plt.ylabel('Probability Density Function')
        plt.show()
        plotName = os.path.join(plotDir,'cl.pdf')
        plt.savefig(plotName,dpi=200)
        plt.close('all')
    
        plt.figure(figsize=(12,10))
        bins1, hist1 = hist_results(r)
        plt.plot(bins1, hist1)
        plt.xlabel('Powerlaw Index')
        plt.ylabel('Probability Density Function')
        plt.show()
        plotName = os.path.join(plotDir,'r.pdf')
        plt.savefig(plotName,dpi=200)
        plt.close('all')

        ns = np.linspace(0.01,2.0,50)
        cls = np.linspace(0.5,1.0,50)

        images_detected = np.zeros((len(ns),len(cls)))
        images_T = np.zeros((len(ns),len(cls)))

        for ii,n in enumerate(ns):
            for jj,cl in enumerate(cls):
                image_array = numimages(prob_data,distmu_data,data_out,Telescope_T,Telescope_m,n,cl,0.0)
                image_array_sorted = np.sort(image_array)

                tmax = tmax = data_out["fade"]*3600
                image_detected = float(len(np.where(image_array < tmax)[0]))/float(len(image_array))

                index = np.floor(len(image_array)*0.5)
                image_T = image_array_sorted[index]

                images_detected[ii,jj] = image_detected
                images_T[ii,jj] = image_T/3600.0

        NS,CLS = np.meshgrid(ns,cls)

        vmin = np.min(images_detected)
        vmax = np.max(images_detected)
        plotName = os.path.join(plotDir,'images_detected.png')
        plt.figure()
        plt.pcolor(NS,CLS,images_detected.T, vmin = vmin, vmax = vmax, cmap = plt.get_cmap('rainbow'))
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Percentage of detections')
        plt.xlabel('Power Law Index')
        plt.ylabel('Confidence Level')
        plt.xlim([0.01,2.0])
        plt.ylim([0.5,1.0])
        plt.show()
        plt.savefig(plotName,dpi=200)
        plotName = os.path.join(plotDir,'images_detected.eps')
        plt.savefig(plotName,dpi=200)
        plotName = os.path.join(plotDir,'images_detected.pdf')
        plt.savefig(plotName,dpi=200)
        plt.close('all')

        vmin = 0.0
        vmax = 10.0
        plotName = os.path.join(plotDir,'images_T.png')
        plt.figure()
        plt.pcolor(NS,CLS,images_T.T, vmin = vmin, vmax = vmax, cmap = plt.get_cmap('rainbow'))
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Median detection time [hours]')
        plt.xlabel('Power Law Index')
        plt.ylabel('Confidence Level')
        plt.xlim([0.01,2.0])
        plt.ylim([0.5,1.0])
        plt.show()
        plt.savefig(plotName,dpi=200)
        plotName = os.path.join(plotDir,'images_T.eps')
        plt.savefig(plotName,dpi=200)
        plotName = os.path.join(plotDir,'images_T.pdf')
        plt.savefig(plotName,dpi=200)
        plt.close('all')

    nums = np.arange(0,len(image_nominal_array_sorted))
    nums[:] = 1.0
    nums = nums / np.sum(nums)
    nums_cumsum = np.cumsum(nums)

    plotName = os.path.join(plotDir,'optimization.png')
    plt.figure()
    plt.semilogx(image_nominal_array_sorted,100*nums_cumsum,'k--',label='Naive')
    plt.semilogx(image_optimal_array_sorted,100*nums_cumsum,'r.-',label='Optimal')
    plt.semilogx(image_known_array_sorted,100*nums_cumsum,'c',label='Known')
    plt.legend(loc=2)
    plt.xlim([10,1e6])
    plt.ylim([0,100.0])
    plt.xlabel('Time [s]')
    plt.ylabel('Percentage of imaged counterparts')
    plt.show()
    plt.savefig(plotName,dpi=200)
    plotName = os.path.join(plotDir,'optimization.eps')
    plt.savefig(plotName,dpi=200)
    plotName = os.path.join(plotDir,'optimization.pdf')
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    filename = os.path.join(plotDir,'images.dat')
    fid = open(filename,'w+')
    for i, j, k, l in zip(nums_cumsum,image_nominal_array_sorted,image_optimal_array_sorted,image_known_array_sorted):
        fid.write('%.5e %.5e %.5e %.5e\n'%(i,j,k,l))
    fid.close()

def mkdir(path):
    """@create path (if it does not already exist).

    @param path
        directory path to create
    """

    pathSplit = path.split("/")
    pathAppend = "/"
    for piece in pathSplit:
        if piece == "":
            continue
        pathAppend = os.path.join(pathAppend,piece)
        if not os.path.isdir(pathAppend):
            os.mkdir(pathAppend)

# Parse command line
opts = parse_commandline()
telescope = opts.telescope

outputDir = opts.outputDir
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)
baseplotDir = os.path.join(outputDir,telescope)
if not os.path.isdir(baseplotDir):
    os.mkdir(baseplotDir)

dataDir = opts.dataDir

directories = glob.glob(os.path.join(dataDir,'*'))

minabsm = opts.minabsm
maxabsm = opts.maxabsm

minfade = opts.minfade
maxfade = opts.maxfade

if telescope == "PS1":
    nside = 16
    # PS1
    # 7 sq degrees (3 deg diameter) 
    #for the i-band, which we nominally use 
    #for t=45, 
    # m_limit (5 sigma) = 21.5  (AB mag) 
    # Using nside=32 which is 3.3 square degrees
    # Using nside=16 which is 13.4 square degrees

    Telescope_T = 45.0
    Telescope_m = 21.5
    Telescope_T = Telescope_T* hp.nside2pixarea(nside, degrees=True) / 7.0 
elif telescope == "ATLAS":

    # ATLAS  
    # 29.2 sq degrees
    # T = 30s
    # m_limit = 18.7 (5 sigma) in the cyan band
    # Using nside=16 which is 13.4 square degrees

    nside = 16
    Telescope_T = 30.0
    Telescope_m = 18.7
    Telescope_T = Telescope_T* hp.nside2pixarea(nside, degrees=True) / 29.2
elif telescope == "combined":
    nside = 16

    PS1_Telescope_T = 45.0
    PS1_Telescope_m = 21.5
    PS1_Telescope_T = PS1_Telescope_T*hp.nside2pixarea(nside, degrees=True) / 7.0

    ATLAS_Telescope_T = 30.0
    ATLAS_Telescope_m = 18.7
    ATLAS_Telescope_T = ATLAS_Telescope_T* hp.nside2pixarea(nside, degrees=True) / 29.2

print "Pixel area: %.4f square degrees" % hp.nside2pixarea(nside, degrees=True)

# http://iopscience.iop.org/article/10.3847/0004-637X/825/1/52/pdf
#5*(log10(400*1e6)-1) = 38.0103
# 28 to 22 = -10 to -16

for directory in directories:
    directorySplit = directory.split("/")[-1]
    plotDir = os.path.join(baseplotDir,directorySplit)

    if not opts.event == None:
        if not opts.event == directorySplit: continue
    print plotDir
    print directorySplit

    filename = os.path.join(directory,'bayestar.fits.gz')
 
    if not os.path.isfile(filename):
        continue

    filename = os.path.join(directory,'lalinference_nest/skymap.fits.gz')
    filename_samples = os.path.join(directory,'lalinference_nest/posterior_samples.dat')

    lines = [line.rstrip('\n') for line in open(filename_samples)]
    data = np.loadtxt(filename_samples,skiprows=1)
    line = lines[0]
    params = line.split("\t")
    params = filter(None, params)
    samples = {}

    for ii in xrange(len(params)):
        param = params[ii]
        samples[param] = data[:,ii]

        print "%s: %.5f"%(param,np.median(samples[param]))

    data_out = {}
    data_out["ra"] = []
    data_out["dec"] = []
    data_out["dist"] = []
    for ii in xrange(opts.nsignals):
        idx = np.floor(np.random.rand()*len(samples["ra"]))
        data_out["ra"].append(samples["ra"][idx])
        data_out["dec"].append(samples["dec"][idx])
        data_out["dist"].append(samples["dist"][idx])    
    data_out["ra"] = np.array(data_out["ra"])
    data_out["dec"] = np.array(data_out["dec"])
    data_out["dist"] = np.array(data_out["dist"])

    plotDir = "%s/%s/%d-%d/%.2f-%.2f"%(baseplotDir,directorySplit,minabsm,maxabsm,minfade,maxfade)
    mkdir(plotDir)
    healpix(filename,plotDir,data_out)

