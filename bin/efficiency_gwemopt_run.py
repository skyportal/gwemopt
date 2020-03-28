import os, sys
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize']=24
plt.rcParams['ytick.labelsize']=24
from matplotlib.pyplot import cm

from matplotlib.patches import Rectangle

configDirectory = "/home/michael.coughlin/gwemopt/config"


events = ["S200105ae","S200213t","S200115j"]
gpstimes = [1262276684.0572,1265602259.327981,1263122607.742047]
skymaps = ["/home/michael.coughlin/gwemopt/data/S200105ae/LALInference.fits.gz","/home/michael.coughlin/gwemopt/data/S200213t/LALInference.fits.gz","/home/michael.coughlin/gwemopt/data/S200115j/LALInference.fits.gz"]

events = ["S200105ae","S200213t"]
gpstimes = [1262276684.0572,1265602259.327981]
skymaps = ["/home/michael.coughlin/gwemopt/data/S200105ae/LALInference.fits.gz","/home/michael.coughlin/gwemopt/data/S200213t/LALInference.fits.gz"]

lightcurveFiles = ["/home/michael.coughlin/gwemopt/lightcurves/Bulla_mejdyn0.005_mejwind0.050_phi45_45.6.dat","/home/michael.coughlin/gwemopt/lightcurves/Bulla_mejdyn0.005_mejwind0.010_phi45_45.6.dat","Tophat","Tophat"]
lightcurveLabels = ['KN: 0.005 $M_\odot$, 0.05 $M_\odot$', 
                    'KN: 0.005 $M_\odot$, 0.01 $M_\odot$',
                    'Tophat: -16 mag, 0 mag/day',
                    'Tophat: -16 mag, 0.5 mag/day']

modelTypes = ["file","file","Tophat","Tophat"]
mags = [-16.0,-16.0,-16.0,-16.0]
dmags = [0.0,0.0,0.0,0.5]
exposuretimes = np.arange(30,630,30)

baseplotDir = "/home/michael.coughlin/gwemopt/output/efficiency"
condorDir = "/home/michael.coughlin/gwemopt/condor"

data = {}

for event,skymap,gpstime in zip(events,skymaps,gpstimes):
    data[event] = {}
    for modelType,lightcurveFile,mag,dmag,lightcurveLabel in zip(modelTypes,lightcurveFiles,mags,dmags,lightcurveLabels):
        for exposuretime in exposuretimes:
            if modelType == "Tophat":
                lcurve = "%.1f_%.1f"%(mag,dmag)
            else:
                lcurve = lightcurveFile.split("/")[-1].replace(".dat","")
            outputDir = os.path.join(baseplotDir,event,lcurve,"%d"%(exposuretime))
            if not lcurve in data[event]:
                data[event][lcurve] = {}
                data[event][lcurve]["label"] = lightcurveLabel
                data[event][lcurve]["ntiles"] = []
                data[event][lcurve]["probs"] = []
                data[event][lcurve]["data"] = []

            efficiency_file = os.path.join(outputDir,'efficiency.txt')
            if os.path.isfile(efficiency_file):
                lines = [line.rstrip('\n') for line in open(efficiency_file,'r')]
                line = lines[1].split("\t")
                data[event][lcurve]["data"].append([exposuretime,float(line[4])])

            schedule_file = os.path.join(outputDir,'coverage_ZTF.dat')
            if os.path.isfile(schedule_file):
                data_out = np.atleast_2d(np.loadtxt(schedule_file))
                if data_out.size == 0: continue
                data[event][lcurve]["ntiles"].append([exposuretime,len(data_out[:,0])])
                data[event][lcurve]["probs"].append([exposuretime,np.sum(data_out[:,1])])
        data[event][lcurve]["ntiles"] = np.array(data[event][lcurve]["ntiles"])
        data[event][lcurve]["probs"] = np.array(data[event][lcurve]["probs"])
        data[event][lcurve]["data"] = np.array(data[event][lcurve]["data"])

color2 = 'coral'
color1 = 'cornflowerblue'
color3 = 'darkgreen'
color4 = 'darkmagenta'

colors = [color1,color2,color3,color4]
linestyles = ['-', '-.', ':','--']

for event,skymap,gpstime in zip(events,skymaps,gpstimes):
    outputDir = os.path.join(baseplotDir,event)
    plt.figure(figsize=(10,6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    for ii, lcurve in enumerate(data[event].keys()):
        if data[event][lcurve]["data"].size == 0: continue
        ax1.step(data[event][lcurve]["data"][:,0], data[event][lcurve]["data"][:,1], linestyles[ii], where='mid', label=data[event][lcurve]["label"], color=colors[ii])
    ax1.legend(loc=9,ncol=2,labelspacing=0.3,columnspacing=0.3,handlelength=2,prop={'size':18})
    for lcurve in data[event].keys():
        ax2.step(data[event][lcurve]["ntiles"][:,0], data[event][lcurve]["probs"][:,1], '-', where='mid',label=data[event][lcurve]["label"], color='k')

    ax1.set_xlabel('Exposure time [s]')
    ax1.set_ylabel('Detection Efficiency')
    ax2.set_ylabel('Integrated Probability')

    ax1.plot([30,30],[0,1],'k--')
    rect1 = Rectangle((120, 0), 180, 1, alpha=0.3, color='g')
    ax1.add_patch(rect1)

    plt.text(40, 1.05, "Survey")
    plt.annotate('', xy=(0.02, 1.03), xytext=(0.17,1.03), xycoords='axes fraction',
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 )

    plt.text(185, 1.05, "ToO")
    plt.annotate('', xy=(0.17, 1.03), xytext=(0.5,1.03), xycoords='axes fraction',
                 arrowprops=dict(facecolor='black', arrowstyle='<->'),
                 )

    ax1.set_xlim([15,600])
    ax1.set_ylim([0,1.0])
    ax2.set_ylim([0,1.0])

    plotName = os.path.join(outputDir,'efficiency.pdf')
    plt.savefig(plotName,bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,6))
    for lcurve in data[event].keys():
        if data[event][lcurve]["ntiles"].size == 0: continue
        plt.step(data[event][lcurve]["ntiles"][:,0], data[event][lcurve]["ntiles"][:,1], '.-', where='mid',label=data[event][lcurve]["label"])
    plt.legend(loc=3)
    plotName = os.path.join(outputDir,'ntiles.pdf')
    plt.savefig(plotName,bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,6))
    for lcurve in data[event].keys():
        if data[event][lcurve]["probs"].size == 0: continue
        plt.step(data[event][lcurve]["probs"][:,0], data[event][lcurve]["probs"][:,1], 'x', label=data[event][lcurve]["label"])
    plt.legend(loc=3)
    plotName = os.path.join(outputDir,'probs.pdf')
    plt.savefig(plotName,bbox_inches='tight')
    plt.close()
