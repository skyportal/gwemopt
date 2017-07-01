
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

def moc(params,map_struct,moc_structs):

    plotName = os.path.join(params["outputDir"],'moc.pdf')
    ax = plt.gca()
    hp.mollview(map_struct["prob"],title='Probability')
    ax = plt.gca()
    for telescope in moc_structs:
        moc_struct = moc_structs[telescope]
        for index in moc_struct.iterkeys():
            ipix, moc, corners, patch = moc_struct[index]["ipix"],  moc_struct[index]["moc"], moc_struct[index]["corners"], moc_struct[index]["patch"]
            #hp.visufunc.projplot(corners[:,0], corners[:,1], 'k', lonlat = True)
            hp.projaxes.HpxMollweideAxes.add_patch(ax,patch)
            #moc.plot()
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

def skymap(params,map_struct):

    plotName = os.path.join(params["outputDir"],'prob.pdf')
    hp.mollview(map_struct["prob"],title='Probability')
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    if "distmu" in map_struct:
        plotName = os.path.join(params["outputDir"],'dist.pdf')
        hp.mollview(map_struct["distmu"],title='Probability',min=0.0,max=100.0)
        plt.show()
        plt.savefig(plotName,dpi=200)
        plt.close('all')

def strategy(params, detmaps, t_detmaps, strategy_struct):

    moviedir = os.path.join(params["outputDir"],'movie')
    if not os.path.isdir(moviedir): os.mkdir(moviedir)
        
    for ii in xrange(len(t_detmaps)):
        t_detmap = t_detmaps[ii]
        detmap = detmaps[ii]

        plotName = os.path.join(moviedir,'detmap-%04d.png'%ii)
        title = "Detectability Map: %.2f Days"%t_detmap
        hp.mollview(detmap,title=title,min=0.0,max=1.0,unit="Probability of Detection")
        plt.show()
        plt.savefig(plotName,dpi=200)
        plt.close('all')

    moviefiles = os.path.join(moviedir,"detmap-%04d.png")
    filename = os.path.join(params["outputDir"],"detmap.mpg")
    ffmpeg_command = 'ffmpeg -an -y -r 20 -i %s -b:v %s %s'%(moviefiles,'5000k',filename)
    os.system(ffmpeg_command)
    filename = os.path.join(params["outputDir"],"detmap.gif")
    ffmpeg_command = 'ffmpeg -an -y -r 20 -i %s -b:v %s %s'%(moviefiles,'5000k',filename)
    os.system(ffmpeg_command)    
    rm_command = "rm %s/*.png"%(moviedir)
    os.system(rm_command)

    plotName = os.path.join(params["outputDir"],'strategy.pdf')
    hp.mollview(strategy_struct,title="Time Allocation",unit="Time [Hours]")
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

def efficiency(params, map_struct, coverage_struct, efficiency_structs):

    plotName = os.path.join(params["outputDir"],'efficiency.pdf')
    for key in efficiency_structs:
        efficiency_struct = efficiency_structs[key]
        plt.loglog(efficiency_struct["distances"],efficiency_struct["efficiency"],label=efficiency_struct["legend_label"])
    plt.xlabel('Distance [Mpc]')
    plt.ylabel('Efficiency')
    plt.legend(loc="best")
    plt.ylim([0.01,1])
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    plotName = os.path.join(params["outputDir"],'injs.pdf')
    plt.plot(efficiency_struct["ra"],efficiency_struct["dec"],'kx')
    plt.xlabel('RA [Degrees]')
    plt.ylabel('Declination [Degrees]')
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    plotName = os.path.join(params["outputDir"],'mollview_injs.pdf')
    hp.mollview(map_struct["prob"])
    hp.projplot(efficiency_struct["ra"], efficiency_struct["dec"], 'wx', lonlat=True, coord='G')
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    plotName = os.path.join(params["outputDir"],'mollview_coverage.pdf')
    hp.mollview(map_struct["prob"])
    hp.projplot(coverage_struct["data"][:,0], coverage_struct["data"][:,1], 'wx', lonlat=True, coord='G')
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

