
import os, sys, copy
import numpy as np
import healpy as hp
import gwemopt.coverage
from astropy.time import Time
import gwemopt.utils

from scipy.stats import norm

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from gwemopt.segments import angular_distance

try:
    import ligo.skymap.plot
    cmap = "cylon"
except:
    cmap = 'PuBuGn'

def observability(params,map_struct):

    observability_struct = map_struct["observability"]

    unit='Gravitational-wave probability'
    cbar=False

    for telescope in observability_struct.keys():
        plotName = os.path.join(params["outputDir"],'observability_%s.pdf'%telescope)
        hp.mollview(map_struct["prob"]*observability_struct[telescope]["observability"],title='',unit=unit,cbar=cbar,min=np.min(map_struct["prob"]),max=np.max(map_struct["prob"]),cmap=cmap)
        add_edges()
        plt.show()
        plt.savefig(plotName,dpi=200)
        plt.close('all')

    if params["doMovie"]:
        moviedir = os.path.join(params["outputDir"],'movie')
        if not os.path.isdir(moviedir): os.mkdir(moviedir)
    
        for telescope in observability_struct.keys():
            dts = list(observability_struct[telescope]["dts"].keys())
            dts = np.sort(dts)
            for ii,dt in enumerate(dts):
                plotName = os.path.join(moviedir,'observability-%04d.png'%ii)
                title = "Detectability Map: %.2f Days"%dt
                hp.mollview(map_struct["prob"]*observability_struct[telescope]["dts"][dt],title=title,cbar=cbar,min=np.min(map_struct["prob"]),max=np.max(map_struct["prob"]),cmap=cmap)
                add_edges()
                plt.show()
                plt.savefig(plotName,dpi=200)
                plt.close('all')
    
            moviefiles = os.path.join(moviedir,"observability-%04d.png")
            filename = os.path.join(params["outputDir"],"observability_%s.mpg"%telescope)
            ffmpeg_command = 'ffmpeg -an -y -r 20 -i %s -b:v %s %s'%(moviefiles,'5000k',filename)
            os.system(ffmpeg_command)
            filename = os.path.join(params["outputDir"],"observability_%s.gif"%telescope)
            ffmpeg_command = 'ffmpeg -an -y -r 20 -i %s -b:v %s %s'%(moviefiles,'5000k',filename)
            os.system(ffmpeg_command)
            rm_command = "rm %s/*.png"%(moviedir)
            os.system(rm_command)
    
def tauprob(params,tau,prob):

    plotName = os.path.join(params["outputDir"],'tau_prob.pdf')
    plt.figure()
    plt.plot(tau, prob)
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Log of observing time $\tau$',fontsize = 20)
    plt.ylabel('Log of detection prob. given the target is at the observing field',fontsize = 20)
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

def tiles(params,map_struct,tiles_structs):

    unit='Gravitational-wave probability'
    cbar=False

    plotName = os.path.join(params["outputDir"],'tiles.pdf')
    plt.figure()
    hp.mollview(map_struct["prob"],title='',unit=unit,cbar=cbar,cmap=cmap)
    ax = plt.gca()
    for telescope in tiles_structs:
        tiles_struct = tiles_structs[telescope]
        for index in tiles_struct.keys():
            ipix, corners, patch = tiles_struct[index]["ipix"], tiles_struct[index]["corners"], tiles_struct[index]["patch"]
            #hp.visufunc.projplot(corners[:,0], corners[:,1], 'k', lonlat = True)
            if not patch: continue
            hp.projaxes.HpxMollweideAxes.add_patch(ax,patch)
            #tiles.plot()
    add_edges()
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    if params["doReferences"]:

        rot=(90,0,0)
        plotName = os.path.join(params["outputDir"],'tiles_ref.pdf')
        plt.figure()
        hp.mollview(np.zeros(map_struct["prob"].shape),title='',unit=unit,cbar=cbar,cmap=cmap)
        ax = plt.gca()
        for telescope in tiles_structs:
            config_struct = params["config"][telescope]
            tiles_struct = tiles_structs[telescope]
            if not "reference_images" in config_struct: continue
            for index in tiles_struct.keys():
                if not index in config_struct["reference_images"]: continue
                if len(params["filters"]) == 1:
                    if not params["filters"][0] in config_struct["reference_images"][index]:
                        continue
                ipix, corners, patch = tiles_struct[index]["ipix"], tiles_struct[index]["corners"], tiles_struct[index]["patch"]
                #hp.visufunc.projplot(corners[:,0], corners[:,1], 'k', lonlat = True)
                if not patch: continue
                patch_cpy = copy.copy(patch)
                patch_cpy.axes = None
                patch_cpy.figure = None
                patch_cpy.set_transform(ax.transData)
                hp.projaxes.HpxMollweideAxes.add_patch(ax,patch_cpy)
                #tiles.plot()
        add_edges()
        plt.show()
        plt.savefig(plotName,dpi=200)
        plt.close('all')

def add_edges():

    hp.graticule(verbose=False)
    plt.grid(True)
    lons = np.arange(-150.0,180,30.0)
    lats = np.zeros(lons.shape)
    for lon, lat in zip(lons,lats):
        hp.projtext(lon,lat,"%.0f"%lon,lonlat=True)
    lats = np.arange(-60.0,90,30.0)
    lons = np.zeros(lons.shape)
    for lon, lat in zip(lons,lats):
        hp.projtext(lon,lat,"%.0f"%lat,lonlat=True)

def skymap(params,map_struct):

    unit='Gravitational-wave probability'
    cbar=False

    lons = np.arange(-150.0,180,30.0)
    lats = np.zeros(lons.shape)

    plotName = os.path.join(params["outputDir"],'prob.pdf')
    if np.percentile(map_struct["prob"],99) > 0:
        hp.mollview(map_struct["prob"],title='',unit=unit,cbar=cbar,min=np.percentile(map_struct["prob"],1),max=np.percentile(map_struct["prob"],99),cmap=cmap)
    else:
        hp.mollview(map_struct["prob"],title='',unit=unit,cbar=cbar,min=np.percentile(map_struct["prob"],1),cmap=cmap)
    add_edges()
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    if "distmu" in map_struct:
        plotName = os.path.join(params["outputDir"],'dist.pdf')
        hp.mollview(map_struct["distmu"],unit='Distance [Mpc]',min=np.percentile(map_struct["distmu"],10),max=np.percentile(map_struct["distmu"],90))
        add_edges()
        plt.show()
        plt.savefig(plotName,dpi=200)
        plt.close('all')

        plotName = os.path.join(params["outputDir"],'dist_median.pdf')
        hp.mollview(map_struct["distmed"],unit='Distance [Mpc]',min=np.percentile(map_struct["distmed"],10),max=np.percentile(map_struct["distmed"],90))
        add_edges()
        plt.show()
        plt.savefig(plotName,dpi=200)
        plt.close('all')

def waw(params, detmaps, t_detmaps, strategy_struct):

    if params["doMovie"]:
        moviedir = os.path.join(params["outputDir"],'movie')
        if not os.path.isdir(moviedir): os.mkdir(moviedir)
        
        for ii in range(len(t_detmaps)):
            t_detmap = t_detmaps[ii]
            detmap = detmaps[ii]

            plotName = os.path.join(moviedir,'detmap-%04d.png'%ii)
            title = "Detectability Map: %.2f Days"%t_detmap
            hp.mollview(detmap,title=title,min=0.0,max=1.0,unit="Probability of Detection")
            add_edges()
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

def efficiency(params, map_struct, efficiency_structs):

    unit='Gravitational-wave probability'
    cbar=False

    plotName = os.path.join(params["outputDir"],'efficiency.pdf')
    plt.figure()
    for key in efficiency_structs:
        efficiency_struct = efficiency_structs[key]
        plt.plot(efficiency_struct["distances"],efficiency_struct["efficiency"],label=efficiency_struct["legend_label"])
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
    hp.mollview(map_struct["prob"],unit=unit,cbar=cbar)
    hp.projplot(efficiency_struct["ra"], efficiency_struct["dec"], 'wx', lonlat=True, coord='G')
    add_edges()
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

def coverage(params, map_struct, coverage_struct, catalog_struct=None):

    unit='Gravitational-wave probability'
    cbar=False

    plotName = os.path.join(params["outputDir"],'mollview_coverage.pdf')
    plt.figure()
    hp.mollview(map_struct["prob"],title='',unit=unit,cbar=cbar,cmap=cmap)
    hp.projplot(coverage_struct["data"][:,0], coverage_struct["data"][:,1], 'wx', lonlat=True, coord='G')
    add_edges()
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    idx = np.isfinite(coverage_struct["data"][:,4])
    if not idx.size: return
    min_time = np.min(coverage_struct["data"][idx,4])
    max_time = np.max(coverage_struct["data"][idx,4])

    plotName = os.path.join(params["outputDir"],'coverage.pdf')
    plt.figure(figsize=(10,8))
    ax = plt.gca()
    for ii in range(len(coverage_struct["ipix"])):
        data = coverage_struct["data"][ii,:]
        filt = coverage_struct["filters"][ii]
        ipix = coverage_struct["ipix"][ii]
        patch = coverage_struct["patch"][ii]
        FOV = coverage_struct["FOV"][ii]

        if filt=="g":
            color = "g"
        elif filt=="r":
            color = "r"
        else:
            color = "k"

        plt.scatter(data[2],data[5],s=20,color=color)

    plt.xlabel("Time [MJD]")
    plt.ylabel("Tile Number")
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    plotName = os.path.join(params["outputDir"],'tiles_coverage.pdf')
    plt.figure()
    hp.mollview(map_struct["prob"],title='',unit=unit,cbar=cbar, cmap=cmap)
    add_edges()
    ax = plt.gca()
    for ii in range(len(coverage_struct["ipix"])):
        data = coverage_struct["data"][ii,:]
        filt = coverage_struct["filters"][ii]
        ipix = coverage_struct["ipix"][ii]
        patch = coverage_struct["patch"][ii]
        FOV = coverage_struct["FOV"][ii]

        #hp.visufunc.projplot(corners[:,0], corners[:,1], 'k', lonlat = True)
        if patch == []:
            continue

        patch_cpy = copy.copy(patch)
        patch_cpy.axes = None
        patch_cpy.figure = None
        patch_cpy.set_transform(ax.transData)
        hp.projaxes.HpxMollweideAxes.add_patch(ax,patch_cpy)
        #tiles.plot()
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    diffs = []
    if params["tilesType"] == "galaxy":
        coverage_ras = coverage_struct["data"][:,0]
        coverage_decs = coverage_struct["data"][:,1]
        coverage_mjds = coverage_struct["data"][:,2]

        for ii in range(len(coverage_ras)-1):
            current_ra, current_dec = coverage_ras[ii], coverage_decs[ii]
            current_mjd = coverage_mjds[ii]

            dist = angular_distance(current_ra, current_dec,
                                    coverage_ras[ii+1:],
                                    coverage_decs[ii+1:])
            idx = np.where(dist <= 1/3600.0)[0]
            if len(idx) > 0:
                jj = idx[0]
                diffs.append(np.abs(coverage_struct["data"][ii,2] - coverage_struct["data"][jj,2]))
    else:
        for ii in range(len(coverage_struct["ipix"])):
            ipix = coverage_struct["ipix"][ii]
            for jj in range(len(coverage_struct["ipix"])):
                if ii >= jj: continue
                if coverage_struct["telescope"][ii] == coverage_struct["telescope"][jj]:
                    continue
                ipix2 = coverage_struct["ipix"][jj]
                overlap = np.intersect1d(ipix, ipix2)
                rat = np.array([float(len(overlap)) / float(len(ipix)),
                                float(len(overlap)) / float(len(ipix2))])
                if np.any(rat > 0.5):
                    diffs.append(np.abs(coverage_struct["data"][ii,2] - coverage_struct["data"][jj,2]))

    filename = os.path.join(params["outputDir"],'tiles_coverage_hist.dat')
    fid = open(filename, 'w')
    for ii in range(len(diffs)):
        fid.write('%.10f\n' % diffs[ii])
    fid.close()

    plotName = os.path.join(params["outputDir"],'tiles_coverage_hist.pdf')
    fig = plt.figure(figsize=(12, 8))
    #hist, bin_edges = np.histogram(diffs, bins=20)
    bins = np.linspace(0.0, 24.0, 25)
    plt.hist(24.0*np.array(diffs), bins=bins)
    plt.xlabel('Difference Between Observations [hours]')
    plt.ylabel('Number of Observations')
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    gpstime = params["gpstime"]
    event_mjd = Time(gpstime, format='gps', scale='utc').mjd

    colors=cm.rainbow(np.linspace(0,1,len(params["telescopes"])))
    plotName = os.path.join(params["outputDir"],'tiles_coverage_int.pdf')

    fig = plt.figure(figsize=(12, 8))

    gs = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(gs[0:3, 0], projection='astro hours mollweide')
    ax2 = fig.add_subplot(gs[3, 0])
    ax3 = ax2.twinx()   # mirror them

    plt.axes(ax1)
    hp.mollview(map_struct["prob"],title='',unit=unit,cbar=cbar, cmap=cmap,
                hold=True)
    add_edges()
    ax = plt.gca()
    data = {}

    if params["tilesType"] == "galaxy":
        for telescope, color in zip(params["telescopes"],colors):
            idx = np.where(coverage_struct["telescope"] == telescope)[0]
            hp.projscatter(coverage_struct["data"][idx,0],
                           coverage_struct["data"][idx,1],
                           lonlat=True, 
                           s=10, color=color)
    else:
        for ii in range(len(coverage_struct["ipix"])):
            data = coverage_struct["data"][ii,:]
            filt = coverage_struct["filters"][ii]
            ipix = coverage_struct["ipix"][ii]
            patch = coverage_struct["patch"][ii]
            FOV = coverage_struct["FOV"][ii]

            idx = params["telescopes"].index(coverage_struct["telescope"][ii])
 
            if patch == []:
                continue
            #hp.visufunc.projplot(corners[:,0], corners[:,1], 'k', lonlat = True)
            patch_cpy = copy.copy(patch)
            patch_cpy.axes = None
            patch_cpy.figure = None
            patch_cpy.set_transform(ax.transData)
            patch_cpy.set_facecolor(colors[idx])

            hp.projaxes.HpxMollweideAxes.add_patch(ax,patch_cpy)
            #tiles.plot()

    idxs = np.argsort(coverage_struct["data"][:,2])
    plt.axes(ax2)
    for telescope, color in zip(params["telescopes"],colors):
        ipixs = np.empty((0,2))
        cum_prob = 0.0
        cum_area = 0.0

        tts, cum_probs, cum_areas = [], [], []
        if params["tilesType"] == "galaxy":
            cum_galaxies = []

        for jj, ii in enumerate(idxs):
            if np.mod(jj, 100) == 0:
                print('%s: %d/%d' % (telescope, jj, len(idxs)))

            data = coverage_struct["data"][ii,:]
            filt = coverage_struct["filters"][ii]
            ipix = coverage_struct["ipix"][ii]
            patch = coverage_struct["patch"][ii]
            FOV = coverage_struct["FOV"][ii]
            area = coverage_struct["area"][ii]
            if params["tilesType"] == "galaxy":
                galaxies = coverage_struct["galaxies"][ii] 

            if not telescope == coverage_struct["telescope"][ii]:
                continue

            if params["tilesType"] == "galaxy":
                overlap = np.setdiff1d(galaxies, cum_galaxies)
                if len(overlap) > 0:
                    for galaxy in galaxies:
                        if galaxy in cum_galaxies: continue
                        if catalog_struct is None: continue
                        if params["galaxy_grade"] == "Sloc":
                            cum_prob = cum_prob + catalog_struct["Sloc"][galaxy]
                        elif params["galaxy_grade"] == "S":
                            cum_prob = cum_prob + catalog_struct["S"][galaxy]
                    cum_galaxies = np.append(cum_galaxies,galaxies)
                    cum_galaxies = np.unique(cum_galaxies).astype(int)
                cum_area = len(cum_galaxies)
            else:
                ipixs = np.append(ipixs,ipix)
                ipixs = np.unique(ipixs).astype(int)

                cum_prob = np.sum(map_struct["prob"][ipixs])
                cum_area = len(ipixs) * map_struct["pixarea_deg2"]

            cum_probs.append(cum_prob)
            cum_areas.append(cum_area)
            tts.append(data[2]-event_mjd)

        ax2.plot(tts, cum_probs, color=color, linestyle='-', label=telescope)
        ax3.plot(tts, cum_areas, color=color, linestyle='--') 

    ax2.set_xlabel('Time since event [days]')
    if params["tilesType"] == "galaxy":
        ax2.set_ylabel('Integrated Metric')
    else:
        ax2.set_ylabel('Integrated Probability')

    if params["tilesType"] == "galaxy":
        ax3.set_ylabel('Number of galaxies')
    else:
        ax3.set_ylabel('Sky area [sq. deg.]')

    ipixs = np.empty((0,2))
    cum_prob = 0.0
    cum_area = 0.0

    tts, cum_probs, cum_areas = [], [], []
    if params["tilesType"] == "galaxy":
        cum_galaxies = []

    for jj, ii in enumerate(idxs):
        data = coverage_struct["data"][ii,:]
        filt = coverage_struct["filters"][ii]
        ipix = coverage_struct["ipix"][ii]
        patch = coverage_struct["patch"][ii]
        FOV = coverage_struct["FOV"][ii]
        area = coverage_struct["area"][ii]
        if params["tilesType"] == "galaxy":
            galaxies = coverage_struct["galaxies"][ii]

        if params["tilesType"] == "galaxy":
            overlap = np.setdiff1d(galaxies, cum_galaxies)
            if len(overlap) > 0:
                for galaxy in galaxies:
                    if galaxy in cum_galaxies: continue
                    if catalog_struct is None: continue
                    if params["galaxy_grade"] == "Sloc":
                        cum_prob = cum_prob + catalog_struct["Sloc"][galaxy]
                    elif params["galaxy_grade"] == "S":
                        cum_prob = cum_prob + catalog_struct["S"][galaxy]
                cum_galaxies = np.append(cum_galaxies,galaxies)
                cum_galaxies = np.unique(cum_galaxies).astype(int)
            cum_area = len(cum_galaxies)
        else:
            ipixs = np.append(ipixs,ipix)
            ipixs = np.unique(ipixs).astype(int)

            cum_prob = np.sum(map_struct["prob"][ipixs])
            cum_area = len(ipixs) * map_struct["pixarea_deg2"]

        tts.append(data[2]-event_mjd)
        cum_probs.append(cum_prob)
        cum_areas.append(cum_area)

    ax2.plot(tts, cum_probs, color='k', linestyle='-', label='All')
    ax3.plot(tts, cum_areas, color='k', linestyle='--')

    if len(params["telescopes"]) > 3:
        ax2.legend(loc=1,ncol=3,fontsize=10)
        ax2.set_ylim([0,1])
        ax3.set_ylim([0,2000])
    elif "IRIS" in params["telescopes"]:
        ax2.set_ylim([0,0.3])
        ax3.set_ylim([0,1200])
        ax2.legend(loc=1)     
    elif "ZTF" in params["telescopes"]:
        ax2.set_ylim([0,0.6])
        ax3.set_ylim([0,6000])
        ax2.legend(loc=1)
    elif "PS1" in params["telescopes"]:
        ax2.set_ylim([0,0.6])
        ax3.set_ylim([0,6000])
        ax2.legend(loc=1)
    else:
        ax2.legend(loc=1)
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    filename = os.path.join(params["outputDir"],'tiles_coverage_int.dat')
    fid = open(filename, 'w')
    for ii in range(len(tts)):
        fid.write('%.10f %.10e %.10f\n' % (tts[ii], cum_probs[ii], cum_areas[ii]))
    fid.close()

    print('Total Cumulative Probability, Area: %.5f, %.5f' % (cum_probs[-1],
                                                              cum_areas[-1]))

    plotName = os.path.join(params["outputDir"],'tiles_coverage_scaled.pdf')
    plt.figure()
    hp.mollview(map_struct["prob"],title='',unit=unit,cbar=cbar,cmap=cmap)
    add_edges()
    ax = plt.gca()
    for ii in range(len(coverage_struct["ipix"])):
        data = coverage_struct["data"][ii,:]
        filt = coverage_struct["filters"][ii]
        ipix = coverage_struct["ipix"][ii]
        patch = coverage_struct["patch"][ii]
        FOV = coverage_struct["FOV"][ii]

        if patch == []:
            continue

        #hp.visufunc.projplot(corners[:,0], corners[:,1], 'k', lonlat = True)
        patch_cpy = copy.copy(patch)
        patch_cpy.axes = None
        patch_cpy.figure = None
        patch_cpy.set_transform(ax.transData)
        current_alpha = patch_cpy.get_alpha()

        if current_alpha > 0.0:
            alpha = data[4]/max_time
            if alpha > 1:
                alpha = 1.0
            patch_cpy.set_alpha(alpha)
        hp.projaxes.HpxMollweideAxes.add_patch(ax,patch_cpy)
        #tiles.plot()
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    if params["doMovie"]:
        idx = np.isfinite(coverage_struct["data"][:,2])
        mjd_min = np.min(coverage_struct["data"][idx,2])
        mjd_max = np.max(coverage_struct["data"][idx,2])
        mjd_N = 100
    
        mjds = np.linspace(mjd_min,mjd_max,num=mjd_N)
        moviedir = os.path.join(params["outputDir"],'movie')
        if not os.path.isdir(moviedir): os.mkdir(moviedir)
    
        #for jj in range(len(coverage_struct["ipix"])):
        #    mjd = coverage_struct["data"][jj,3]
        for jj in range(len(mjds)):
            mjd = mjds[jj]
            plotName = os.path.join(moviedir,'coverage-%04d.png'%jj)
            title = "Coverage Map: %.2f"%mjd
    
            plt.figure()
            hp.mollview(map_struct["prob"],title=title,unit=unit,cbar=cbar,
                        cmap=cmap)
            add_edges()
            ax = plt.gca()
    
            idx = np.where(coverage_struct["data"][:,2]<=mjd)[0]
            #for ii in range(jj):
            for ii in idx: 
                data = coverage_struct["data"][ii,:]
                filt = coverage_struct["filters"][ii]
                ipix = coverage_struct["ipix"][ii]
                patch = coverage_struct["patch"][ii]
                FOV = coverage_struct["FOV"][ii]
    
                if patch == []:
                    continue

                #hp.visufunc.projplot(corners[:,0], corners[:,1], 'k', lonlat = True)
                patch_cpy = copy.copy(patch)
                patch_cpy.axes = None
                patch_cpy.figure = None
                patch_cpy.set_transform(ax.transData)
                #alpha = data[4]/max_time
                #if alpha > 1:
                #    alpha = 1.0
                #patch_cpy.set_alpha(alpha)
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

def scheduler(params,exposurelist,keys): 
    plotName = os.path.join(params["outputDir"],'scheduler.pdf')
    if params['scheduleType'].endswith('_slew'):
        e = []
        k = []
        start_time = exposurelist[0][0]
        for exposure, key in zip(exposurelist, keys):
            e.append((exposure[0] - start_time) * 24)
            k.append(key)
            e.append((exposure[1] - start_time) * 24)
            k.append(key)
        plt.figure()
        plt.grid()
        plt.xlabel("Time (h)")
        plt.ylabel("Tile Number")
        plt.plot(e, k, 'b-')
    else:
        xs = []
        ys = []
        for ii,key in zip(np.arange(len(exposurelist)),keys):
            xs.append(ii)
            ys.append(key)    
        plt.figure()
        plt.xlabel("Exposure Number")
        plt.ylabel("Tile Number")
        plt.plot(xs,ys,'kx')
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

def transients(params, map_struct, transients_struct):

    unit='Gravitational-wave probability'
    cbar=False

    ra = transients_struct["data"][:,0]
    dec = transients_struct["data"][:,1]

    plotName = os.path.join(params["outputDir"],'transients.pdf')
    plt.figure()
    hp.mollview(map_struct["prob"],unit=unit,cbar=cbar)
    hp.projplot(ra, dec, 'wx', lonlat=True, coord='G')
    add_edges()
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

def doMovie_supersched(params,coverage_structs,tile_structs,map_struct):
    
    unit='Gravitational-wave probability'
    cbar=False
    
    idx = np.isfinite(coverage_structs["data"][:,2])
    try:
        mjd_min = np.min(coverage_structs["data"][idx,2])
    except:
        raise ValueError("Not enough scheduled tiles during observation round to make movie.")
    mjd_max = np.max(coverage_structs["data"][idx,2])
    mjd_N = 100
    
    mjds = np.linspace(mjd_min,mjd_max,num=mjd_N)
    
    parentdir = os.path.abspath(os.path.join(params["outputDir"], os.pardir))
    moviedir = os.path.join(parentdir, 'movie') #saves movie file in parent directory
    
    if not os.path.isdir(moviedir): os.mkdir(moviedir)
    
    #for jj in range(len(coverage_struct["ipix"])):
    #    mjd = coverage_struct["data"][jj,3]
    
    for jj in range(len(mjds)):
        mjd = mjds[jj]
        for i in [0,1,2]: #can change this to enumerate(Tobs) if Tobs is variable
            ii = jj+(100*i)
            if not os.path.exists(os.path.join(moviedir,f'coverage-{ii:04d}.png')): #adds multiples of 100 for each round of Tobs
                plotName = os.path.join(moviedir,f'coverage-{ii:04d}.png')
                break
    
        title = "Coverage Map: %.2f"%mjd
        plt.figure()
        hp.mollview(map_struct["prob"],title=title,unit=unit,cbar=cbar,cmap=cmap)
        add_edges()
        ax = plt.gca()
        
        idx = np.where(coverage_structs["data"][:,2]<=mjd)[0]
        
        #for ii in range(jj):
        for ii in idx:
            data = coverage_structs["data"][ii,:]
            filt = coverage_structs["filters"][ii]
            ipix = coverage_structs["ipix"][ii]
            patch = coverage_structs["patch"][ii]
            FOV = coverage_structs["FOV"][ii]
            
            if patch == []:
                continue
        
            #hp.visufunc.projplot(corners[:,0], corners[:,1], 'k', lonlat = True)
            patch_cpy = copy.copy(patch)
            patch_cpy.axes = None
            patch_cpy.figure = None
            patch_cpy.set_transform(ax.transData)
            
            #alpha = data[4]/max_time
            #if alpha > 1:
            #    alpha = 1.0
            #patch_cpy.set_alpha(alpha)
            hp.projaxes.HpxMollweideAxes.add_patch(ax,patch_cpy)
        #tiles.plot()

        Tobs = list(params["Tobs_all"])
        Tobs = np.linspace(Tobs[0],Tobs[1],params["Tobs_split"]+1)

        if f'{Tobs[0]:.2}_to_{Tobs[1]:.2}' not in params["outputDir"]: #only proceeds if not first round of Tobs
            #if single:
                
            for i,Tob in enumerate(Tobs[1:]):
                if f'{Tobs[i]:.2}_to_{Tobs[i+1]:.2}' in params["outputDir"]:
                    break
        
            while i>0: #goes through all previous rounds
                readfile = os.path.join(parentdir,f'{Tobs[i-1]:.2}_to_{Tobs[i]:.2}_Tobs')
                prevtelescopes = params["alltelescopes"][i-1].split(",")
                prev_tile_structs = params["tile_structs"][f'tile_structs_{i-1}']
                i-=1
                for prevtelescope in prevtelescopes:
                    prev_tile_struct = prev_tile_structs[prevtelescope]
                    schedfile = f'schedule_{prevtelescope}.dat'
                    data_file = os.path.join(readfile,schedfile)
                
                    with open(data_file, "r") as f:
                        for line in f:
                            data = list(line.split(' '))
                            field_id = int(data[0])
                            if int(data[8]) == 1:
                                patch = prev_tile_struct[field_id]["patch"]
                                if patch == []:
                                    continue
                                patch_cpy = copy.copy(patch)
                                patch_cpy.axes = None
                                patch_cpy.figure = None
                                patch_cpy.set_transform(ax.transData)
                                patch_cpy.set_facecolor('white')
                                hp.projaxes.HpxMollweideAxes.add_patch(ax,patch_cpy)
    
        plt.show()
        plt.savefig(plotName,dpi=200)
        plt.close('all')
