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

configDirectory = "/home/michael.coughlin/gwemopt/config"


events = ["S200105ae","S200213t","S200115j"]
gpstimes = [1262276684.0572,1265602259.327981,1263122607.742047]
skymaps = ["/home/michael.coughlin/gwemopt/data/S200105ae/LALInference.fits.gz","/home/michael.coughlin/gwemopt/data/S200213t/LALInference.fits.gz","/home/michael.coughlin/gwemopt/data/S200115j/LALInference.fits.gz"]

lightcurveFiles = ["/home/michael.coughlin/gwemopt/lightcurves/Bulla_mejdyn0.005_mejwind0.110_phi45_45.6.dat","/home/michael.coughlin/gwemopt/lightcurves/Bulla_mejdyn0.005_mejwind0.010_phi45_45.6.dat","Tophat","Tophat"]
lightcurveLabels = ['KN: 0.005, 0.11', 'KN: 0.005, 0.01', 'Tophat: 16, 0',
                    'Tophat: 16, 0.5']

modelTypes = ["file","file","Tophat","Tophat"]
mags = [-16.0,-16.0,-16.0,-16.0]
dmags = [0.0,0.0,0.0,0.5]
exposuretimes = np.arange(60,3660,60)

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
                data[event][lcurve]["data"] = []

            efficiency_file = os.path.join(outputDir,'efficiency.txt')
            if os.path.isfile(efficiency_file):
                lines = [line.rstrip('\n') for line in open(efficiency_file,'r')]
                line = lines[1].split("\t")
                data[event][lcurve]["data"].append([exposuretime,float(line[4])])
        data[event][lcurve]["data"] = np.array(data[event][lcurve]["data"])

for event,skymap,gpstime in zip(events,skymaps,gpstimes):
    outputDir = os.path.join(baseplotDir,event)
    plt.figure(figsize=(10,6))
    for lcurve in data[event].keys():
        if data[event][lcurve]["data"].size == 0: continue
        plt.plot(data[event][lcurve]["data"][:,0], data[event][lcurve]["data"][:,1], 'x', label=data[event][lcurve]["label"])
    plt.legend(loc=3)
    plotName = os.path.join(outputDir,'efficiency.pdf')
    plt.savefig(plotName,bbox_inches='tight')
    plt.close()
