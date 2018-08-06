
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

def observability(params,map_struct):
    observability_struct = map_struct["observability"]

    unit='Gravitational-wave probability'
    cbar=False

    for telescope in observability_struct.keys():
        plotName = os.path.join(params["outputDir"],'observability_%s.pdf'%telescope)
        hp.mollview(map_struct["prob"]*observability_struct[telescope]["observability"],title='',unit=unit,cbar=cbar,min=np.min(map_struct["prob"]),max=np.max(map_struct["prob"]))
        add_edges()
        plt.show()
        plt.savefig(plotName,dpi=200)
        plt.close('all')

    if params["doMovie"]:
        moviedir = os.path.join(params["outputDir"],'movie')
        if not os.path.isdir(moviedir): os.mkdir(moviedir)
    
        for telescope in observability_struct.keys():
            dts = observability_struct[telescope]["dts"].keys()
            dts = np.sort(dts)
            for ii,dt in enumerate(dts):
                plotName = os.path.join(moviedir,'observability-%04d.png'%ii)
                title = "Detectability Map: %.2f Days"%dt
                hp.mollview(map_struct["prob"]*observability_struct[telescope]["dts"][dt],title=title,cbar=cbar,min=np.min(map_struct["prob"]),max=np.max(map_struct["prob"]))
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
    hp.mollview(map_struct["prob"],title='',unit=unit,cbar=cbar)
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
        hp.mollview(np.zeros(map_struct["prob"].shape),title='',unit=unit,cbar=cbar,cmap=plt.get_cmap('Greens'))
        ax = plt.gca()
        for telescope in tiles_structs:
            config_struct = params["config"][telescope]
            tiles_struct = tiles_structs[telescope]
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
    hp.mollview(map_struct["prob"],title='',unit=unit,cbar=cbar,min=np.percentile(map_struct["prob"],1),max=np.percentile(map_struct["prob"],99))
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

def coverage(params, map_struct, coverage_struct):

    unit='Gravitational-wave probability'
    cbar=False

    plotName = os.path.join(params["outputDir"],'mollview_coverage.pdf')
    plt.figure()
    hp.mollview(map_struct["prob"],title='',unit=unit,cbar=cbar)
    hp.projplot(coverage_struct["data"][:,0], coverage_struct["data"][:,1], 'wx', lonlat=True, coord='G')
    add_edges()
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    idx = np.isfinite(coverage_struct["data"][:,4])
    if not idx.size: return
    min_time = np.min(coverage_struct["data"][idx,4])
    max_time = np.max(coverage_struct["data"][idx,4])

    plotName = os.path.join(params["outputDir"],'tiles_coverage.pdf')
    plt.figure()
    hp.mollview(map_struct["prob"],title='',unit=unit,cbar=cbar)
    add_edges()
    ax = plt.gca()
    for ii in range(len(coverage_struct["ipix"])):
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
        hp.projaxes.HpxMollweideAxes.add_patch(ax,patch_cpy)
        #tiles.plot()
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    plotName = os.path.join(params["outputDir"],'tiles_coverage_scaled.pdf')
    plt.figure()
    hp.mollview(map_struct["prob"],title='',unit=unit,cbar=cbar)
    add_edges()
    ax = plt.gca()
    for ii in range(len(coverage_struct["ipix"])):
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
            hp.mollview(map_struct["prob"],title=title,unit=unit,cbar=cbar)
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

