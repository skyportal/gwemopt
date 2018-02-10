
import os, sys
import time
import copy

import numpy as np
import healpy as hp

import gwemopt.utils

def hierarchical_tiles_struct(params, config_struct, telescope, map_struct):

    import pymultinest

    Ntiles = params["Ntiles"]

    map_struct_copy = copy.deepcopy(map_struct)

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

        if prob == 0:
            prob = -np.inf

        #if np.isfinite(prob):
        #    print(ra, dec, prob)

        return prob

    def myloglike_circle(cube, ndim, nparams):
        ra = cube[0]
        dec = cube[1]

        ipix, radecs, patch, area = gwemopt.utils.getCirclePixels(ra, dec, config_struct["FOV"], nside)

        if len(ipix) == 0:
            prob = -np.inf
        else:
            prob = np.sum(map_struct_copy["prob"][ipix])

        if prob == 0:
            prob = -np.inf

        if np.isfinite(prob):
            print(ra, dec, prob)

        return prob

    plotDir = os.path.join(params["outputDir"],"multinest")
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)

    nside = params["nside"]
    tile_struct = {}
    for ii in range(Ntiles):

        parameters = ["ra","dec"]
        n_params = len(parameters)
        if config_struct["FOV_type"] == "square":
             pymultinest.run(myloglike_square, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = False, sampling_efficiency = 'parameter', n_live_points = 1000, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = 0.05, multimodal = False, seed = 1)
        elif config_struct["FOV_type"] == "circle":
             pymultinest.run(myloglike_circle, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = False, sampling_efficiency = 'parameter', n_live_points = 1000, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = 0.5, multimodal = False, seed = 1)

        multifile= os.path.join(plotDir,'2-.txt')
        data = np.loadtxt(multifile)
        loglikelihood = -(1/2.0)*data[:,1]
        idx = np.argmax(loglikelihood)
        ra_pointing = data[idx,2]
        dec_pointing = data[idx,3]

        tile_struct[ii] = {}

        if config_struct["FOV_type"] == "square":
            ipix, radecs, patch, area = gwemopt.utils.getSquarePixels(ra_pointing, dec_pointing, config_struct["FOV"], nside, alpha=0.8)
        elif config_struct["FOV_type"] == "circle":
            ipix, radecs, patch, area = gwemopt.utils.getCirclePixels(ra_pointing, dec_pointing, config_struct["FOV"], nside, alpha=0.8)

        tile_struct[ii]["ra"] = ra_pointing
        tile_struct[ii]["dec"] = dec_pointing
        tile_struct[ii]["ipix"] = ipix
        tile_struct[ii]["corners"] = radecs
        tile_struct[ii]["patch"] = patch
        tile_struct[ii]["area"] = area

        map_struct_copy["prob"][ipix] = 0.0
        os.system("rm %s/*"%plotDir)

    return tile_struct

def greedy_tiles_struct(params, config_struct, telescope, map_struct, Ntiles = 10):

    Ntiles = params["Ntiles"]

    map_struct_copy = copy.deepcopy(map_struct)
    skymapData = []
    skymapData.append(map_struct_copy["ra"])
    skymapData.append(map_struct_copy["dec"])
    skymapData.append(map_struct_copy["prob"])
    greedy_struct = PlaceTile(skymapData, config_struct, numtiles = Ntiles)
    tile_struct = greedy_struct.getSamples()

    return tile_struct   

def getRandomPos(ra, dec, nums=1):
    '''
    Return one or more random ra and dec from the list of ras and decs supplied
    pattern of return array [RA, RA, RA, Dec, Dec, Dec] for nums=3
    '''
    return np.hstack([np.random.choice(ra, nums), np.random.choice(dec, nums)])


class PlaceTile:
    def __init__(self, skymapData, config_struct, numtiles=2):
        self.skymapData = skymapData
        self.ra_map = self.skymapData[0]
        self.dec_map = self.skymapData[1]
        self.pVal = self.skymapData[2]
        self.config_struct = config_struct
        self.numtiles = numtiles

    def lnprior(self, skyposition): ### Basic uniform prior across the sky
        reshaped_skyposition = np.reshape(skyposition, (2, self.numtiles))
        ra_centers = reshaped_skyposition[0]
        dec_centers= reshaped_skyposition[1]

        ignore = (ra_centers > 360.) | (ra_centers < 0.) | (dec_centers > 90.) | (dec_centers < -90.)
        if np.sum(ignore):
            return -np.inf
        else:
            return 0
 

    def lnlikelihood(self, skyposition_cent):
        reshaped_skyposition = np.reshape(skyposition_cent, (2, self.numtiles))

        config_struct = self.config_struct
        pVal = copy.deepcopy(self.pVal)
        npix = len(pVal)
        nside = hp.npix2nside(npix) 

        ra_centers = reshaped_skyposition[0]
        dec_centers= reshaped_skyposition[1]
        probabilitySum = 0.0
        for ii in range(self.numtiles):
            if config_struct["FOV_type"] == "square":
                ipix, radecs, patch, area = gwemopt.utils.getSquarePixels(ra_centers[ii], dec_centers[ii], config_struct["FOV"], nside)
            elif config_struct["FOV_type"] == "circle":
                ipix, radecs, patch, area = gwemopt.utils.getCirclePixels(ra_centers[ii], dec_centers[ii], config_struct["FOV"], nside)

            prob = np.sum(pVal[ipix])
            pVal[ipix] = 0.0

            probabilitySum += prob

        return 4*np.log(probabilitySum)


    def lnpost(self, skypos):
        lp = self.lnprior(skypos)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlikelihood(skypos)
        
        
    
    def getSamples(self):

        import emcee        

        config_struct = self.config_struct
        pVal = copy.copy(self.pVal)
        npix = len(pVal)
        nside = hp.npix2nside(npix)

        ndim, nwalkers = 2*self.numtiles, 4*self.numtiles
        include = np.cumsum(self.pVal) < 0.9
        #include[np.sum(include)] = True
        ra_included = self.ra_map[include]
        dec_included = self.dec_map[include]
        p0 = []
        [p0.append(getRandomPos(self.ra_map, self.dec_map, nums=self.numtiles)) for _ in range(nwalkers)]
                    
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnpost)
        start = time.time()
        pos, prob, state = sampler.run_mcmc(p0, 100)
        print('Burned in...')
        print('Time taken to burn in = ' + str(time.time() - start))
        sampler.reset()
        start = time.time()
        result = sampler.run_mcmc(pos, 1000)
        end = time.time()
        print('Acceptance fraction = ' + str(np.mean(sampler.acceptance_fraction[:])))
        print('Time taken to finish the MCMC = ' + str(end - start))
        tileCenters = sampler.flatchain
        samples    = tileCenters.copy()

        peak_ras, peak_decs = self.localizeTC(samples=samples)

        tile_struct = {}
        #for ii in range(self.numtiles):
        #    ra_pointing = samples[0,ii]
        #    dec_pointing = samples[0,ii+self.numtiles]
        for ii in range(len(peak_ras)):
            ra_pointing = peak_ras[ii]
            dec_pointing = peak_decs[ii]

            tile_struct[ii] = {}

            if config_struct["FOV_type"] == "square":
                ipix, radecs, patch, area = gwemopt.utils.getSquarePixels(ra_pointing, dec_pointing, config_struct["FOV"], nside, alpha=0.8)
            elif config_struct["FOV_type"] == "circle":
                ipix, radecs, patch, area = gwemopt.utils.getCirclePixels(ra_pointing, dec_pointing, config_struct["FOV"], nside, alpha=0.8)

        tile_struct[ii]["ra"] = ra_pointing
        tile_struct[ii]["dec"] = dec_pointing
        tile_struct[ii]["ipix"] = ipix
        tile_struct[ii]["corners"] = radecs
        tile_struct[ii]["patch"] = patch
        tile_struct[ii]["area"] = area
    
        return tile_struct
        
        
    def optimizeBins(self, ra, dec, masked_points=np.array([])):
        config_struct = self.config_struct

        pVal = copy.deepcopy(self.pVal)
        if len(masked_points) > 0:
            pVal[masked_points.astype(int)] = 0.0
        npix = len(pVal)
        nside = hp.npix2nside(npix)
 
        binTrials = int(np.sqrt(len(ra)))
        bins_array = np.arange(2, binTrials + 1)
        probs_allChosenTiles = np.array([])
    
        for bins in bins_array:

            [hist, ra_bin, dec_bin] = np.histogram2d(ra, dec, bins)
            ra_bin_cent = 0.5*(ra_bin[1:] + ra_bin[:-1])
            dec_bin_cent = 0.5*(dec_bin[1:] + dec_bin[:-1])
            max_pos =np.argmax(hist)
            x = int(np.argmax(hist)/bins)
            y = np.argmax(hist)%bins
            ra_peak = ra_bin_cent[x]
            dec_peak = dec_bin_cent[y]

            if config_struct["FOV_type"] == "square":
                ipix, radecs, patch, area = gwemopt.utils.getSquarePixels(ra_peak, dec_peak, config_struct["FOV"], nside)
            elif config_struct["FOV_type"] == "circle":
                ipix, radecs, patch, area = gwemopt.utils.getCirclePixels(ra_peak, dec_peak, config_struct["FOV"], nside)

            prob_encl = np.sum(pVal[ipix])
            probs_allChosenTiles = np.append(probs_allChosenTiles, prob_encl)
        
        bin_max = bins_array[np.argmax(probs_allChosenTiles)]

        [hist_max, ra_bin_max, dec_bin_max] = np.histogram2d(ra, dec, bin_max)
        ra_bin_cent = 0.5*(ra_bin_max[1:] + ra_bin_max[:-1])
        dec_bin_cent = 0.5*(dec_bin_max[1:] + dec_bin_max[:-1])

    
        y = np.argmax(hist_max)%bin_max
        x = int(np.argmax(hist_max)/bin_max)
        ra_peak = ra_bin_cent[x]
        dec_peak = dec_bin_cent[y]

        if config_struct["FOV_type"] == "square":
            ipix, radecs, patch, area = gwemopt.utils.getSquarePixels(ra_peak, dec_peak, config_struct["FOV"], nside)
        elif config_struct["FOV_type"] == "circle":
            ipix, radecs, patch, area = gwemopt.utils.getCirclePixels(ra_peak, dec_peak, config_struct["FOV"], nside)
    
        return ra_peak, dec_peak, ipix

    def localizeTC(self, reference=None, samples=None, verbose=False):
    
        if samples is None:
            samples = self.getSamples()
        else:
            print('\n\nReading data from pickled files...')

        ra_samples = []
        dec_samples = []
        for ii in range(0, self.numtiles):
            ra_samples.append(samples[:,ii])
            dec_samples.append(samples[:,ii+self.numtiles])

        masked_points=np.array([])
        probabilitySum = 0.0
        RA_Peak_list = []
        Dec_Peak_list = []
        for ii in range(self.numtiles):
            ra_peak, dec_peak, points_to_be_masked = self.optimizeBins(ra_samples[ii], dec_samples[ii], masked_points)

            masked_points = np.hstack((masked_points, points_to_be_masked))
            RA_Peak_list.append(ra_peak)
            Dec_Peak_list.append(dec_peak)

        return [RA_Peak_list, Dec_Peak_list]


    
    









