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

import os, sys, glob, optparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
        "font.size": 24.0,
        "axes.titlesize": 24.0,
        "axes.labelsize": 24.0,
        "xtick.labelsize": 24.0,
        "ytick.labelsize": 24.0,
        "legend.fontsize": 24.0,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "text.usetex": True
})

__author__  = "Michael Coughlin <michael.coughlin@ligo.org>"
__author__  = "Duncan Meacher <duncan.meacher@ligo.org>"
__date__    = "2016/01/06"
__version__ = "0.1"

def parse_commandline():
    """@parse the options given on the command-line.
    """
    parser = optparse.OptionParser(usage=__doc__,version=__version__)

    parser.add_option("-o","--outputDir",help="Output location",default = '/home/mcoughlin/Skymaps/optimization/plots')

    parser.add_option("--minabsm", help="Min absolute magnitude.",default=-16,type=float)
    parser.add_option("--maxabsm", help="Max absolute magnitude.",default=-10,type=float)
    parser.add_option("--minfade", help="Min time above detection threshold [hours].",default=0.1,type=float)
    parser.add_option("--maxfade", help="Max time above detection threshold [hours].",default=10,type=float)

    parser.add_option("-v", "--verbose", action="store_true", default=False,help="Run verbosely. (Default: False)")

    opts, args = parser.parse_args()

    # show parameters
    if opts.verbose:
        print >> sys.stderr, ""
        print >> sys.stderr, "running dL.py..."
        print >> sys.stderr, "version: %s"%__version__
        print >> sys.stderr, ""
        print >> sys.stderr, "***************** PARAMETERS ********************"
        for o in opts.__dict__.items():
          print >> sys.stderr, o[0]+":"
          print >> sys.stderr, o[1]
        print >> sys.stderr, ""

    return opts

def combine_params(runpath):
    folders = glob.glob(os.path.join(runpath,"*"))
    for ii,folder in enumerate(folders):
        #baseDir = os.path.join(folder,"-16--10")
        baseDir = "%s/%d-%d/%.2f-%.2f"%(folder,minabsm,maxabsm,minfade,maxfade)
        #resultsfile = os.path.join(baseDir,"best.dat")
        resultsfile = os.path.join(baseDir,"samples.dat")
        if not os.path.isfile(resultsfile): continue
        data_out = np.loadtxt(resultsfile)
        if ii == 0:
            params = data_out
        else:
            params = np.vstack((params,data_out))
    return params

def combine_results(runpath):

    folders = glob.glob(os.path.join(runpath,"*"))
    for ii,folder in enumerate(folders):
        #baseDir = os.path.join(folder,"-16--10")
        baseDir = "%s/%d-%d/%.2f-%.2f"%(folder,minabsm,maxabsm,minfade,maxfade)

        resultsfile = os.path.join(baseDir,"images.dat")
        if not os.path.isfile(resultsfile): continue
        nums_cumsum, image_array = load_results(baseDir)
        if ii == 0:
            image_arrays = image_array
        else:
            image_arrays = np.vstack((image_arrays,image_array))
    image_arrays = image_arrays.T
    for ii,col in enumerate(image_arrays):
        image_arrays[ii,:] = np.sort(col)

    idx_10 = int(np.floor(0.10*len(col)))
    idx_50 = int(np.floor(0.50*len(col)))
    idx_90 = int(np.floor(0.90*len(col)))

    image_10 = image_arrays[:,idx_10]
    image_50 = image_arrays[:,idx_50]
    image_90 = image_arrays[:,idx_90]

    times_interp = np.logspace(0,6,1000)
    image_10_interp = np.interp(times_interp,image_10,nums_cumsum)
    image_50_interp = np.interp(times_interp,image_50,nums_cumsum)
    image_90_interp = np.interp(times_interp,image_90,nums_cumsum)

    return nums_cumsum,image_10,image_50,image_90,times_interp,image_10_interp,image_50_interp,image_90_interp

def load_results(baseDir):

    resultsfile = os.path.join(baseDir,"images.dat")
    data_out = np.loadtxt(resultsfile)
    nums_cumsum = data_out[:,0]
    image_optimal_array_sorted = data_out[:,2] 

    nums_cumsum_all = np.linspace(0.0,1.0,1000)
 
    image_array = np.interp(nums_cumsum_all,nums_cumsum,image_optimal_array_sorted)
    image_array[np.isnan(image_array)] = np.inf

    return nums_cumsum_all, image_array

def hist_results(samples):

    bins = np.linspace(np.min(samples),np.max(samples),50)
    hist1, bin_edges = np.histogram(samples, bins=bins)
    hist1 = hist1 / float(np.sum(hist1))
    bins = (bins[1:] + bins[:-1])/2.0

    return bins, hist1

# =============================================================================
#
#                                    MAIN
#
# =============================================================================

# Parse command line
opts = parse_commandline()

minabsm = opts.minabsm
maxabsm = opts.maxabsm

minfade = opts.minfade
maxfade = opts.maxfade

outputDir = opts.outputDir 
resultsDir = os.path.join(outputDir,'results')
if not os.path.isdir(resultsDir):
    os.mkdir(resultsDir)

runpath = os.path.join(outputDir,'PS1')
nums_cumsum,ps1_10,ps1_50,ps1_90,times_interp,ps1_10_interp,ps1_50_interp,ps1_90_interp = combine_results(runpath)
ps1_params = combine_params(runpath)
runpath = os.path.join(outputDir,'ATLAS')
nums_cumsum,atlas_10,atlas_50,atlas_90,times_interp,atlas_10_interp,atlas_50_interp,atlas_90_interp = combine_results(runpath)
atlas_params = combine_params(runpath)
runpath = os.path.join(outputDir,'combined')
nums_cumsum,combined_10,combined_50,combined_90,times_interp,combined_10_interp,combined_50_interp,combined_90_interp = combine_results(runpath)
combined_params = combine_params(runpath)

plt.figure(figsize=(12,10))
plt.semilogx(ps1_50,100*nums_cumsum,'g--',label='PS1')
plt.semilogx(atlas_50,100*nums_cumsum,'r.-',label='ATLAS')
plt.semilogx(combined_50,100*nums_cumsum,'c',label='PS1/ATLAS')
plt.fill_between(times_interp,100*ps1_90_interp,100*ps1_10_interp,facecolor='green',alpha=0.5)
plt.fill_between(times_interp,100*atlas_90_interp,100*atlas_10_interp,facecolor='red',alpha=0.5)
plt.fill_between(times_interp,100*combined_90_interp,100*combined_10_interp,facecolor='cyan',alpha=0.5)
#plt.semilogx(ps1_10,100*nums_cumsum,'k--')
#plt.semilogx(ps1_90,100*nums_cumsum,'k--')
#plt.semilogx(atlas_10,100*nums_cumsum,'r.-')
#plt.semilogx(atlas_90,100*nums_cumsum,'r.-')
#plt.semilogx(combined_10,100*nums_cumsum,'c')
#plt.semilogx(combined_90,100*nums_cumsum,'c')
plt.legend(loc=2)
plt.xlim([10,1e6])
plt.ylim([0,100.0])
plt.xlabel('Time [s]')
plt.ylabel('Percentage of imaged counterparts')
plt.show()
plotName = os.path.join(resultsDir,'optimization.pdf')
plt.savefig(plotName,dpi=200)
plt.close('all')

plt.figure(figsize=(12,10))
bins1, hist1 = hist_results(ps1_params[:,0])
bins2, hist2 = hist_results(combined_params[:,0])
bins3, hist3 = hist_results(atlas_params[:,0])
bins4, hist4 = hist_results(combined_params[:,3])
#plt.plot(bins1, np.cumsum(hist1),'k',label='PS1')
#plt.plot(bins2, np.cumsum(hist2),'k--',label='PS1 with ATLAS')
#plt.plot(bins3, np.cumsum(hist3),'r',label='ATLAS')
#plt.plot(bins4, np.cumsum(hist4),'r--',label='ATLAS with PS1')
plt.plot(bins1, hist1,'k',label='PS1')
plt.plot(bins2, hist2,'k--',label='PS1 with ATLAS')
plt.plot(bins3, hist3,'r',label='ATLAS')
plt.plot(bins4, hist4,'r--',label='ATLAS with PS1')
plt.legend(loc='best')
#plt.xlim([10,1e6])
#plt.ylim([0,1.0])
plt.xlabel('Likelihood Powerlaw Index')
plt.ylabel('Probability Density Function')
plt.show()
plotName = os.path.join(resultsDir,'n.pdf')
plt.savefig(plotName,dpi=200)
plt.close('all')

plt.figure(figsize=(12,10))
bins1, hist1 = hist_results(ps1_params[:,1])
bins2, hist2 = hist_results(combined_params[:,1])
bins3, hist3 = hist_results(atlas_params[:,1])
bins4, hist4 = hist_results(combined_params[:,4])
#plt.plot(bins1, np.cumsum(hist1),'k',label='PS1')
#plt.plot(bins2, np.cumsum(hist2),'k--',label='PS1 with ATLAS')
#plt.plot(bins3, np.cumsum(hist3),'r',label='ATLAS')
#plt.plot(bins4, np.cumsum(hist4),'r--',label='ATLAS with PS1')
plt.plot(bins1, hist1,'k',label='PS1')
plt.plot(bins2, hist2,'k--',label='PS1 with ATLAS')
plt.plot(bins3, hist3,'r',label='ATLAS')
plt.plot(bins4, hist4,'r--',label='ATLAS with PS1')
plt.legend(loc='best')
#plt.xlim([10,1e6])
#plt.ylim([0,1.0])
plt.xlabel('Likelihood Confidence Level')
plt.ylabel('Probability Density Function')
plt.show()
plotName = os.path.join(resultsDir,'cl.pdf')
plt.savefig(plotName,dpi=200)
plt.close('all')

plt.figure(figsize=(12,10))
bins1, hist1 = hist_results(ps1_params[:,2])
bins2, hist2 = hist_results(combined_params[:,2])
bins3, hist3 = hist_results(atlas_params[:,2])
bins4, hist4 = hist_results(combined_params[:,5])
#plt.plot(bins1, np.cumsum(hist1),'k',label='PS1')
#plt.plot(bins2, np.cumsum(hist2),'k--',label='PS1 with ATLAS')
#plt.plot(bins3, np.cumsum(hist3),'r',label='ATLAS')
#plt.plot(bins4, np.cumsum(hist4),'r--',label='ATLAS with PS1')
plt.plot(bins1, hist1,'k',label='PS1')
plt.plot(bins2, hist2,'k--',label='PS1 with ATLAS')
plt.plot(bins3, hist3,'r',label='ATLAS')
plt.plot(bins4, hist4,'r--',label='ATLAS with PS1')
plt.legend(loc='best')
#plt.xlim([10,1e6])
#plt.ylim([0,1.0])
plt.xlabel('Distance Powerlaw Index')
plt.ylabel('Probability Density Function')
plt.show()
plotName = os.path.join(resultsDir,'r.pdf')
plt.savefig(plotName,dpi=200)
plt.close('all')
