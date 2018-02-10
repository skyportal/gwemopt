
import os, sys
import optparse
import numpy as np
import glob
import scipy.stats

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

def compute_apparent_magnitude_samples(params, lightcurve_struct, samples_struct, t):

    filt = params["config"][params["telescopes"][0]]["filt"]
    F = np.zeros((len(samples_struct['ra']),len(t)))
    lightcurve_t = lightcurve_struct["t"]
    lightcurve_mag = lightcurve_struct[filt]
    lightcurve_mag_interp = np.interp(t,lightcurve_t,lightcurve_mag)

    for ii in range(len(samples_struct['dist'])):
        F[ii,:] = lightcurve_mag_interp + 5*(np.log10(samples_struct['dist'][ii]*1e6) - 1)

    return F

def compute_apparent_magnitude(params, lightcurve_struct, t):

    filt = params["config"][params["telescopes"][0]]["filt"]
    lightcurve_t = lightcurve_struct["t"]
    lightcurve_mag = lightcurve_struct[filt]
    lightcurve_mag_interp = np.interp(t,lightcurve_t,lightcurve_mag)

    return lightcurve_mag_interp

def read_files_lbol(files,tmin=-100.0,tmax=100.0):

    names = []
    Lbols = {}
    for filename in files:
        name = filename.replace(".txt","").replace(".dat","").split("/")[-1]
        Lbol_d = np.loadtxt(filename)
        #Lbol_d = Lbol_d[1:,:]

        t = Lbol_d[:,0]
        Lbol = Lbol_d[:,1]
        try:
            index = np.nanargmin(Lbol)
            index = 0
        except:
            index = 0
        t0 = t[index]

        Lbols[name] = {}
        Lbols[name]["t"] = Lbol_d[:,0]
        indexes1 = np.where(Lbols[name]["t"]>=tmin)[0]
        indexes2 = np.where(Lbols[name]["t"]<=tmax)[0]
        indexes = np.intersect1d(indexes1,indexes2)

        Lbols[name]["t"] = Lbol_d[indexes,0]
        Lbols[name]["Lbol"] = Lbol_d[indexes,1]

        names.append(name)

    return Lbols, names

def read_files(files,tmin=-100.0,tmax=100.0):

    names = []
    legend_names = []
    mags = {}
    for filename in files:
        name = filename.replace(".txt","").replace(".dat","").split("/")[-1]

        if "neutron_precursor" in name:
            legend_label = "Barnes et al. (2016)"
        elif "rpft" in name:
            legend_label = "Metzger et al. (2015)"
        elif "BHNS" in name:
            legend_label = "Kawaguchi et al. (2016)"
        elif "BNS" in name:
            legend_label = "Dietrich et al. (2016)"
        elif "k1" in name:
            legend_label = "Tanaka and Hotokezaka (2013)"
        else:
            legend_label = name
       
        mag_d = np.loadtxt(filename)
        #mag_d = mag_d[1:,:]

        t = mag_d[:,0]
        g = mag_d[:,1]
        try:
            index = np.nanargmin(g)
            index = 0
        except:
            index = 0
        t0 = t[index]

        mags[name] = {}
        mags[name]["t"] = mag_d[:,0]
        indexes1 = np.where(mags[name]["t"]>=tmin)[0]
        indexes2 = np.where(mags[name]["t"]<=tmax)[0] 
        indexes = np.intersect1d(indexes1,indexes2)

        mags[name]["t"] = mag_d[indexes,0]
        mags[name]["g"] = mag_d[indexes,1]
        mags[name]["r"] = mag_d[indexes,2]
        mags[name]["i"] = mag_d[indexes,3]
        mags[name]["z"] = mag_d[indexes,4]

        mags[name]["c"] = (mags[name]["g"]+mags[name]["r"])/2.0
        mags[name]["o"] = (mags[name]["r"]+mags[name]["i"])/2.0

        mags[name]["name"] = name
        mags[name]["legend_label"] = legend_label

    return mags

def xcorr_mags(mags1,mags2):
    nmags1 = len(mags1)
    nmags2 = len(mags2)
    xcorrvals = np.zeros((nmags1,nmags2))
    chisquarevals = np.zeros((nmags1,nmags2))
    for ii,name1 in enumerate(mags1.iterkeys()):
        for jj,name2 in enumerate(mags2.iterkeys()):

            t1 = mags1[name1]["t"]
            t2 = mags2[name2]["t"]
            t = np.unique(np.append(t1,t2))
            t = np.arange(-100,100,0.1)      

            mag1 = np.interp(t, t1, mags1[name1]["g"])
            mag2 = np.interp(t, t2, mags2[name2]["g"])

            indexes1 = np.where(~np.isnan(mag1))[0]
            indexes2 = np.where(~np.isnan(mag2))[0]
            indexes = np.intersect1d(indexes1,indexes2)
            mag1 = mag1[indexes1]
            mag2 = mag2[indexes2]

            indexes1 = np.where(~np.isinf(mag1))[0]
            indexes2 = np.where(~np.isinf(mag2))[0]
            indexes = np.intersect1d(indexes1,indexes2)
            mag1 = mag1[indexes1]
            mag2 = mag2[indexes2]

            if len(indexes) == 0:
                xcorrvals[ii,jj] = 0.0
                continue           

            if len(mag1) < len(mag2):
                mag1vals = (mag1 - np.mean(mag1)) / (np.std(mag1) * len(mag1))
                mag2vals = (mag2 - np.mean(mag2)) / (np.std(mag2))
            else:
                mag1vals = (mag1 - np.mean(mag1)) / (np.std(mag1))
                mag2vals = (mag2 - np.mean(mag2)) / (np.std(mag2) * len(mag2))

            xcorr = np.correlate(mag1vals, mag2vals, mode='full')
            xcorr_corr = np.max(np.abs(xcorr))

            #mag1 = mag1 * 100.0 / np.sum(mag1)
            #mag2 = mag2 * 100.0 / np.sum(mag2)

            nslides = len(mag1) - len(mag1)
            if nslides == 0:
                chisquares = scipy.stats.chisquare(mag1, f_exp=mag1)[0]
            elif nslides > 0:
                chisquares = []
                for kk in range(np.abs(nslides)):
                    chisquare = scipy.stats.chisquare(mag1, f_exp=mag2[kk:len(mag1)])[0] 
                    chisquares.append(chisquare)
            elif nslides < 0:
                chisquares = []
                for kk in range(np.abs(nslides)):
                    chisquare = scipy.stats.chisquare(mag2, f_exp=mag1[kk:len(mag2)])[0] 
                    chisquares.append(chisquare)

            print(name1, name2, xcorr_corr, np.min(np.abs(chisquares)), len(mag1), len(mag2))
            xcorrvals[ii,jj] = xcorr_corr
            chisquarevals[ii,jj] = np.min(np.abs(chisquares))

    return xcorrvals, chisquarevals
