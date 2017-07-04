
import os, sys, copy
import numpy as np
import healpy as hp

from scipy.stats import norm

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.pyplot as plt

def tiles(params,map_struct,tiles_structs):

    plotName = os.path.join(params["outputDir"],'tiles.pdf')
    plt.figure()
    hp.mollview(map_struct["prob"],title='Probability')
    ax = plt.gca()
    for telescope in tiles_structs:
        tiles_struct = tiles_structs[telescope]
        for index in tiles_struct.iterkeys():
            ipix, corners, patch = tiles_struct[index]["ipix"], tiles_struct[index]["corners"], tiles_struct[index]["patch"]
            #hp.visufunc.projplot(corners[:,0], corners[:,1], 'k', lonlat = True)
            hp.projaxes.HpxMollweideAxes.add_patch(ax,patch)
            #tiles.plot()
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
    plt.figure()
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
    plt.figure()
    plt.plot(efficiency_struct["ra"],efficiency_struct["dec"],'kx')
    plt.xlabel('RA [Degrees]')
    plt.ylabel('Declination [Degrees]')
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    plotName = os.path.join(params["outputDir"],'mollview_injs.pdf')
    plt.figure()
    hp.mollview(map_struct["prob"])
    hp.projplot(efficiency_struct["ra"], efficiency_struct["dec"], 'wx', lonlat=True, coord='G')
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    plotName = os.path.join(params["outputDir"],'mollview_coverage.pdf')
    plt.figure()
    hp.mollview(map_struct["prob"])
    hp.projplot(coverage_struct["data"][:,0], coverage_struct["data"][:,1], 'wx', lonlat=True, coord='G')
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    n_windows = len(params["Tobs"]) // 2
    tot_obs_time = np.sum(np.diff(params["Tobs"])[::2]) * 86400.

    min_time = 0.0
    #max_time = 10.0*config_struct["exposuretime"]
    max_time = 3600.0

    plotName = os.path.join(params["outputDir"],'mollview_tiles_coverage.pdf')
    plt.figure()
    hp.mollview(map_struct["prob"],title='Probability')
    ax = plt.gca()
    for ii in xrange(len(coverage_struct["ipix"])):
        data = coverage_struct["data"][ii,:]
        filt = coverage_struct["filters"][ii]
        ipix = coverage_struct["ipix"][ii]
        patch = coverage_struct["patch"][ii]
        FOV = coverage_struct["FOV"][ii]

        #hp.visufunc.projplot(corners[:,0], corners[:,1], 'k', lonlat = True)
        patch_cpy = copy.copy(patch)
        patch_cpy.axes = None
        patch_cpy.figure = None
        patch_cpy.set_transform(ax.transData)
        alpha = data[4]/max_time
        if alpha > 1:
            alpha = 1.0
        patch_cpy.set_alpha(alpha)
        hp.projaxes.HpxMollweideAxes.add_patch(ax,patch_cpy)
        #tiles.plot()
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    moviedir = os.path.join(params["outputDir"],'movie')
    if not os.path.isdir(moviedir): os.mkdir(moviedir)

    for jj in xrange(len(coverage_struct["ipix"])):
        mjd = coverage_struct["data"][jj,3]
        plotName = os.path.join(moviedir,'coverage-%04d.png'%jj)
        title = "Coverage Map: %.2f"%mjd       

        plt.figure()
        hp.mollview(map_struct["prob"],title=title)
        ax = plt.gca()
        for ii in xrange(jj):
            data = coverage_struct["data"][ii,:]
            filt = coverage_struct["filters"][ii]
            ipix = coverage_struct["ipix"][ii]
            patch = coverage_struct["patch"][ii]
            FOV = coverage_struct["FOV"][ii]

            #hp.visufunc.projplot(corners[:,0], corners[:,1], 'k', lonlat = True)
            patch_cpy = copy.copy(patch)
            patch_cpy.axes = None
            patch_cpy.figure = None
            patch_cpy.set_transform(ax.transData)
            alpha = data[4]/max_time
            if alpha > 1:
                alpha = 1.0
            patch_cpy.set_alpha(alpha)
            hp.projaxes.HpxMollweideAxes.add_patch(ax,patch_cpy)
            #tiles.plot()
        plt.show()
        plt.savefig(plotName,dpi=200)
        plt.close('all')
        
    moviefiles = os.path.join(moviedir,"coverage-%04d.png")
    filename = os.path.join(params["outputDir"],"coverage.mpg")
    ffmpeg_command = 'ffmpeg -an -y -r 20 -i %s -b:v %s %s'%(moviefiles,'5000k',filename)
    os.system(ffmpeg_command)
    filename = os.path.join(params["outputDir"],"coverage.gif")
    ffmpeg_command = 'ffmpeg -an -y -r 20 -i %s -b:v %s %s'%(moviefiles,'5000k',filename)
    os.system(ffmpeg_command)
    rm_command = "rm %s/*.png"%(moviedir)
    os.system(rm_command)

