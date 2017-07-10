
import os, sys
import copy

import numpy as np
import scipy.optimize

import gwemopt.utils

import pymultinest

def greedy_tiles_struct(params, config_struct, telescope, map_struct, Ntiles = 50):

    map_struct_copy = copy.copy(map_struct)

    def myprior(cube, ndim, nparams):
        cube[0] = cube[0]*360.0
        cube[1] = cube[1]*180.0 - 90.0

    def myloglike_square(cube, ndim, nparams):
        ra = cube[0]
        dec = cube[1]

        ipix, radecs, patch, area = gwemopt.utils.getSquarePixels(ra, dec, config_struct["FOV"], nside)
        
        if len(ipix) == 0:
            prob = -np.inf
        else:
            prob = np.sum(map_struct_copy["prob"][ipix])

        #if np.isfinite(prob):
        #    print ra, dec, prob

        return prob

    def myloglike_circle(cube, ndim, nparams):
        ra = cube[0]
        dec = cube[1]

        ipix, radecs, patch, area = gwemopt.utils.getCirclePixels(ra, dec, config_struct["FOV"], nside)

        if len(ipix) == 0:
            prob = -np.inf
        else:
            prob = np.sum(map_struct_copy["prob"][ipix])

        if np.isfinite(prob):
            print ra, dec, prob

        return prob

    plotDir = os.path.join(params["outputDir"],"multinest")
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)

    nside = params["nside"]
    tile_struct = {}
    for ii in xrange(Ntiles):

        parameters = ["ra","dec"]
        n_params = len(parameters)
        if config_struct["FOV_type"] == "square":
             pymultinest.run(myloglike_square, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = False, sampling_efficiency = 'parameter', n_live_points = 1000, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = 0.05, multimodal = False)
        elif config_struct["FOV_type"] == "circle":
             pymultinest.run(myloglike_circle, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = False, sampling_efficiency = 'parameter', n_live_points = 1000, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = 0.5, multimodal = False)

        multifile= os.path.join(plotDir,'2-.txt')
        data = np.loadtxt(multifile)
        loglikelihood = -(1/2.0)*data[:,1]
        idx = np.argmax(loglikelihood)
        ra_pointing = data[idx,2]
        dec_pointing = data[idx,3]        

        tile_struct[ii] = {}

        if config_struct["FOV_type"] == "square":
            ipix, radecs, patch, area = gwemopt.utils.getSquarePixels(ra_pointing, dec_pointing, config_struct["FOV"], nside)
        elif config_struct["FOV_type"] == "circle":
            ipix, radecs, patch, area = gwemopt.utils.getCirclePixels(ra_pointing, dec_pointing, config_struct["FOV"], nside)

        tile_struct[ii]["ra"] = ra_pointing
        tile_struct[ii]["dec"] = dec_pointing
        tile_struct[ii]["ipix"] = ipix
        tile_struct[ii]["corners"] = radecs
        tile_struct[ii]["patch"] = patch
        tile_struct[ii]["area"] = area

        map_struct_copy["prob"][ipix] = 0.0
        os.system("rm %s/*"%plotDir)

    return tile_struct
