# Copyright (C) 2017 Shaon Ghosh, David Kaplan, Shasvath Kapadia, Deep Chatterjee
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""

Creates ranked tiles for a given gravitational wave trigger. Sample steps are below:


tileObj = rankedTilesGenerator.RankedTileGenerator('bayestar.fits.gz')
[ranked_tile_index, ranked_tile_probs] = tileObj.getRankedTiles(resolution=512)

This gives the ranked tile indices and their probabilities for the bayestar sky-map.
The resolution is 512, thus ud_grading to this value from the actual sky-map resolution.
The code expects the file ZTF_tiles_set1_nowrap_indexed.dat and the pickled file 
preComputed_pixel_indices_512.dat to be in the same path. 

"""

import os
import numpy as np
import pylab as pl
import pickle
import sys
from math import ceil
import healpy as hp
from scipy import interpolate

import time
import datetime

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import get_sun
from astropy.coordinates import get_moon
from astropy.coordinates import get_body
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

import gwemopt.moc

def create_ranked(params, map_struct):

    nside = params["nside"]

    moc_structs = {}
    for telescope in params["telescopes"]:
        config_struct = params["config"][telescope]
        tesselation = config_struct["tesselation"]

        preComputedFile = os.path.join(params["tilingDir"],'preComputed_%s_pixel_indices_%d.dat'%(telescope,nside))
        if not os.path.isfile(preComputedFile):
            print("Creating tiles file...")
            gwemopt.rankedTilesGenerator.createTileFile(params,preComputedFile,radecs=config_struct["tesselation"])

        preCompDictFiles = {64:None, 128:None,256:None, 512:None, 1024:None, 2048:None}
        preCompDictFiles[nside] = preComputedFile

        tileObj = RankedTileGenerator(map_struct["prob"],preCompDictFiles=preCompDictFiles)

        ranked_tile_index, ranked_tile_probs, ipixs = tileObj.getRankedTiles(resolution=params["nside"])

        moc_struct = {}
        for ii, tess in enumerate(tesselation):
            index, ra, dec = tess[0], tess[1], tess[2]
            index = index.astype(int)
            moc_struct[index] = gwemopt.moc.Fov2Moc(params, config_struct, telescope, ra, dec, nside)
            idx = np.where(ranked_tile_index==ii)[0][0]
            moc_struct[index]["ipix"] = ipixs[idx]

        moc_structs[telescope] = moc_struct

    return moc_structs

def createTileFile(params, preComputedFile, radecs=None, tileFile=None):

    nside = params["nside"]
    npix = hp.nside2npix(nside)

    theta, phi = hp.pix2ang(nside, np.arange(0, npix)) # Construct the theta and phi arrays
    ra = np.rad2deg(phi) # Construct ra array
    dec = np.rad2deg(0.5*np.pi - theta) # Construct dec array
    pixelIndex = np.arange(npix)

    if not tileFile is None:
        data = np.recfromtxt(tileFile, names=True)
        RA_tile = data['ra_center'] ### RA value of the telescope fields
        Dec_tile = data['dec_center'] ### Dec values of the telescope fields
        tile_index = data['ID']-1 ### Indexing the tiles
    elif not radecs is None:
        RA_tile = radecs[:,1]
        Dec_tile = radecs[:,2]
        tile_index = np.arange(len(RA_tile)) + 1.0

    closestTileIndex = []

    ras_all, decs_all = [], []
    for ii in pixelIndex:
            s = np.arccos( np.sin(np.pi*dec[ii]/180.)\
                    * np.sin(np.pi*Dec_tile/180.)\
                    + np.cos(np.pi*dec[ii]/180.)\
                    * np.cos(np.pi*Dec_tile/180.) \
                    * np.cos(np.pi*(RA_tile - ra[ii])/180.) )
            index = np.argmin(s) ### minimum angular distance index

            ras_all.append(RA_tile[index])
            decs_all.append(Dec_tile[index])
            closestTileIndex.append(tile_index[index])

    closestTileIndex = np.array(closestTileIndex)
    ras_all = np.array(ras_all)
    decs_all = np.array(decs_all)
 
    uniqueTiles, indices = np.unique(closestTileIndex, return_index=True)
    ras_unique = ras_all[indices]
    decs_unique = decs_all[indices] 

    pixelsInTile = {}
    for ii, tile in enumerate(uniqueTiles):
        whereThisTile = tile == closestTileIndex ### pixels indices in this tile
        pixelsInTile[ii+1] = [pixelIndex[whereThisTile],ras_unique[ii],decs_unique[ii]]

    File = open(preComputedFile, 'wb')
    pickle.dump(pixelsInTile, File)
    File.close()

def getTileBounds(FOV, ra_cent, dec_cent):
    dec_down = dec_cent - 0.5*np.sqrt(FOV)
    dec_up = dec_cent + 0.5*np.sqrt(FOV)

    ra_down_left = ra_cent - 0.5*(np.sqrt(FOV)/(np.cos(dec_down*(np.pi/180.))))
    ra_down_right = ra_cent + 0.5*(np.sqrt(FOV)/(np.cos(dec_down*(np.pi/180.))))
    ra_up_left = ra_cent - 0.5*(np.sqrt(FOV)/(np.cos(dec_up*(np.pi/180.))))
    ra_up_right = ra_cent + 0.5*(np.sqrt(FOV)/(np.cos(dec_up*(np.pi/180.))))
    
    return([dec_down, dec_up, ra_down_left, ra_down_right, ra_up_left, ra_up_right])



class RankedTileGenerator:
	def __init__(self, skymap, preComputed_64=None, preComputed_128=None, preComputed_256=None, preComputed_512=None, preComputed_1024=None, preComputed_2048=None, preCompDictFiles=None):
		self.skymap = skymap
		npix = len(self.skymap)
		self.nside = hp.npix2nside(npix)

		if preCompDictFiles==None:
			self.preCompDictFiles = {64:preComputed_64, 128:preComputed_128,256:preComputed_256, 512:preComputed_512, 1024:preComputed_1024, 2048:preComputed_2048}
		else:
                        self.preCompDictFiles = preCompDictFiles

	def sourceTile(self, ra, dec, tiles):
		'''
		METHOD     :: This method takes the position of the injected 
					  event and returns the tile index
					  
		ra		   :: Right ascension of the source in degrees
		dec		   :: Declination angle of the source in degrees
		tiles	   :: The tile coordinate file (in the following format)
			      ID	ra_center	dec_center	
			      1  	24.714290	-85.938460
			      2  	76.142860	-85.938460
			      ...
		'''
		tileData = np.recfromtxt(tiles, names=True)
		Dec_tile = tileData['dec_center']
		RA_tile = tileData['ra_center']
		ID = tileData['ID']
		s = np.arccos( np.sin(np.pi*dec/180.)\
			* np.sin(np.pi*Dec_tile/180.)\
			+ np.cos(np.pi*dec/180.)\
			* np.cos(np.pi*Dec_tile/180.) \
			* np.cos(np.pi*(RA_tile - ra)/180.) )
		index = np.argmin(s) ### minimum angular distance index

		return ID[index] - 1 ### Since the indexing begins with 1.

	
	def searchedArea(self, ra, dec, resolution=None):
		'''
		METHOD     :: This method takes the position of the injected 
			      event and the sky-map. It returns the searched 
			      area of the sky-map to reach to the source lo-
			      cation. The searched area constitutes both the
			      total area (sq. deg) that needed to be search-
			      ed to reach the source location, and the total
			      localization probability covered in the process.
					  
		ra		   :: Right ascension of the source in degrees
		dec		   :: Declination angle of the source in degrees
		resolution 	   :: The value of the nside, if not supplied, 
				      the default skymap is used.
		'''
		if not resolution:
			resolution = self.nside
		n = np.log(resolution)/np.log(2)
		resolution = int(2 ** round(n)) ## resolution in powers of 2
		if resolution > 2048: resolution = 2048
		if resolution < 64: resolution = 64
		filename = self.preCompDictFiles[resolution]
		File = open(filename, 'rb')
		data = pickle.load(File)
		tile_index = np.arange(len(data))
		skymapUD = hp.ud_grade(self.skymap, resolution, power=-2)
		npix = len(skymapUD)
		theta, phi = hp.pix2ang(resolution, np.arange(0, npix))
		ra_map = np.rad2deg(phi) # Construct ra array
		dec_map = np.rad2deg(0.5*np.pi - theta) # Construct dec array
		pVal = skymapUD[np.arange(0, npix)]
		order = np.argsort(-pVal)
		ra_map = ra_map[order]
		dec_map = dec_map[order]
		pVal = pVal[order]
		s = np.arccos( np.sin(np.pi*dec/180.)\
			* np.sin(np.pi*dec_map/180.)\
			+ np.cos(np.pi*dec/180.)\
			* np.cos(np.pi*dec_map/180.) \
			* np.cos(np.pi*(ra_map - ra)/180.) )
		index = np.argmin(s) ### minimum angular distance index
		coveredProb = np.sum(pVal[0:index])
		searchedArea = index*hp.nside2pixarea(resolution, degrees=True)
		return [searchedArea, coveredProb]

	
	def getRankedTiles(self, resolution=None, verbose=False):
		'''
		METHOD		:: This method returns two numpy arrays, the first
				   contains the tile indices of telescope and the second
				   contains the probability values of the corresponding 
				   tiles. The tiles are sorted based on their probability 
				   values.
		
		resolution  :: The value of the nside, if not supplied, 
			       the default skymap is used.
		'''
		if not resolution:
			resolution = self.nside
		n = np.log(resolution)/np.log(2)
		resolution = int(2 ** round(n)) ## resolution in powers of 2
		if resolution > 2048: resolution = 2048
		if resolution < 64: resolution = 64
		if verbose: print('Using resolution of ' + str(resolution))
		filename = self.preCompDictFiles[resolution]
		if verbose: print(filename)
		File = open(filename, 'rb')
		data = pickle.load(File)
		tile_index = np.arange(len(data))
		skymapUD = hp.ud_grade(self.skymap, resolution, power=-2)
		npix = len(skymapUD)
		theta, phi = hp.pix2ang(resolution, np.arange(0, npix))
		pVal = skymapUD[np.arange(0, npix)]

		allTiles_ipixs = []
		allTiles_probs = []
		for ii in range(1, len(data)+1):
			ipix, ras, decs = data[ii]
			pTile = np.sum(pVal[ipix])
			allTiles_probs.append(pTile)
			allTiles_ipixs.append(ipix)

		allTiles_probs = np.array(allTiles_probs)
		allTiles_ipixs = np.array(allTiles_ipixs)
		index = np.argsort(-allTiles_probs)

		allTiles_probs_sorted = allTiles_probs[index]
		tile_index_sorted = tile_index[index]
		allTiles_ipixs_sorted = allTiles_ipixs[index]
		
		return [tile_index_sorted, allTiles_probs_sorted, allTiles_ipixs_sorted]


	def plotTiles(self, ranked_tile_indices, allTiles_probs_sorted, tileFile, FOV=None,
				  resolution=None, tileEdges=False, CI=0.9):
		'''
		METHOD 	:: This method plots the ranked-tiles on a hammer projection
				   skymap. 
		ranked_tile_indices    :: The index of he ranked-tiles
		allTiles_probs_sorted  :: The probabilities of the ranked-tiles
		tileFile    :: The file with tile indices and centers
			       ID	ra_center	dec_center	
			       1  	24.714290	-85.938460
			       2  	76.142860	-85.938460
		
		FOV		:: Field of view of the telescopes. If not supplied,
				   tile boundaries will not be plotted.

		resolution  	:: The resolution of the skymap to be used.
		tileEdges	:: Allows plotting of the tile edges. Default is False.
		'''			

		from AllSkyMap_basic import AllSkyMap
		import pylab as pl
		
		skymap = self.skymap
		if resolution:
			skymap = hp.ud_grade(skymap, resolution, power=-2)
		npix = len(skymap)
		nside = hp.npix2nside(npix)
		theta, phi = hp.pix2ang(nside, np.arange(0, npix))
		ra = np.rad2deg(phi)
		dec = np.rad2deg(0.5*np.pi - theta)
		pVal = skymap[np.arange(0, npix)]
		order = np.argsort(-pVal)
		ra = ra[order]
		dec = dec[order]
		pVal = pVal[order]
		include = np.cumsum(pVal) < CI
		include[np.sum(include)] = True
		ra_CI = ra[include]
		dec_CI = dec[include]
		pVal_CI = pVal[include]


		pl.figure(figsize=(80,70))
		pl.rcParams.update({'font.size': 60})
		
		m = AllSkyMap(projection='hammer')
		RAP_map, DecP_map = m(ra_CI, dec_CI) 
		m.drawparallels(np.arange(-90.,120.,20.), color='grey', 
						labels=[False,True,True,False], labelstyle='+/-')
		m.drawmeridians(np.arange(0.,420.,30.), color='grey')
		m.drawmapboundary(fill_color='white')
		lons = np.arange(-150,151,30)
		m.label_meridians(lons, fontsize=60, vnudge=1, halign='left', hnudge=-1) 
		m.plot(RAP_map, DecP_map, 'r.', markersize=3, alpha=0.1) 

		tileData = np.recfromtxt(tileFile, names=True)
		
		Dec_tile = tileData['dec_center']
		RA_tile = tileData['ra_center']
		ID = tileData['ID']
		
		include_tiles = np.cumsum(allTiles_probs_sorted) < CI
		include_tiles[np.sum(include_tiles)] = True
		ranked_tile_indices = ranked_tile_indices[include_tiles]

		if FOV is None:
			tileEdges = False
		for ii in ranked_tile_indices:

			if tileEdges:
				[dec_down, dec_up,
				ra_down_left, ra_down_right, 
				ra_up_left, ra_up_right] = getTileBounds(FOV, RA_tile[ii], Dec_tile[ii])
			
			RAP_peak, DecP_peak = m(RA_tile[ii], Dec_tile[ii])
			
			if tileEdges:
				RAP1, DecP1 = m(ra_up_left, dec_up)
				RAP2, DecP2 = m(ra_up_right, dec_up)
				RAP3, DecP3 = m(ra_down_left, dec_down)
				RAP4, DecP4 = m(ra_down_right, dec_down)

			m.plot(RAP_peak, DecP_peak, 'ko', markersize=20, mew=1)
		
			if tileEdges:
				m.plot([RAP1, RAP2], [DecP1, DecP2],'k-', linewidth=4) 
				m.plot([RAP2, RAP4], [DecP2, DecP4],'k-', linewidth=4) 
				m.plot([RAP4, RAP3], [DecP4, DecP3],'k-', linewidth=4) 
				m.plot([RAP3, RAP1], [DecP3, DecP1],'k-', linewidth=4) 

		pl.show()

	def rankGalaxies2D(self, catalog, resolution=None):
		'''
		METHOD  :: This method takes as input a galaxy catalog pickle file
				   that is generated by running the createCatalog.py script.
				   The output is the IDs of the galaxies from the catalog 
				   ranked based on their localization probability.
		
		catalog	:: A pickle file which stores a 7 col numpy array with. The 
				   columns of this array are defined below:
				   		col1 : galaxy ID
				   		col2 : distance to the galaxy
				   		col3 : Declination angle of the galaxy
				   		col4 : Right ascencion of the galaxy
				   		col5 : Closest BAYESTAR Healpix pixel to the galaxy
				   		col6 : Declination angle of the closest pixel
				   		col7 : Right ascencion of the closest pixel
				   		
		resolution :: Optional argument. allows you to fix the resolution of 
					  the skymap. Currently the catalog file has only been 
					  generated for resolution of 512. Use this value.
				   
		'''

		if not resolution:
			resolution = self.nside
		n = np.log(resolution)/np.log(2)
		resolution = int(2 ** round(n)) ## resolution in powers of 2
		if resolution > 2048: resolution = 2048
		if resolution < 64: resolution = 64
		filename = self.preCompDictFiles[resolution]
		File = open(filename, 'rb')
		data = pickle.load(File)
		tile_index = np.arange(len(data))
		skymapUD = hp.ud_grade(self.skymap, resolution, power=-2)
		npix = len(skymapUD)
		theta, phi = hp.pix2ang(resolution, np.arange(0, npix))
		ra_map = np.rad2deg(phi) # Construct ra array
		dec_map = np.rad2deg(0.5*np.pi - theta) # Construct dec array
		pVal = skymapUD[np.arange(0, npix)]
		
		catalogFile = open(catalog, 'rb')
		catalogData = pickle.load(catalogFile)
		
		indices = catalogData[:,4].astype('int') ### Indices of pixels for all galaxies
		galaxy_probs = pVal[indices] ### Probability values of the galaxies in catalog
		order = np.argsort(-galaxy_probs) ### Sorting in descending order of probability
		galaxy_indices = catalogData[:,0].astype('int') ### Indices of galaxies
		ranked_galaxies = galaxy_indices[order]
		galaxy_probs = galaxy_probs[order]
		
		return [ranked_galaxies, galaxy_probs]

	### Older version ###	

# 	def integrationTime(self, T_obs, pValTiles=None):
# 		'''
# 		METHOD :: This method accepts the probability values of the ranked tiles, the 
# 			  total observation time and the rank of the source tile. It returns 
# 			  the array of time to be spent in each tile which is determined based
# 			  on the localizaton probability of the tile. 
# 				  
# 		pValTiles :: The probability value of the ranked tiles. Obtained from getRankedTiles 
# 					 output
# 		T_obs     :: Total observation time available for the follow-up.
# 		'''
# 		if pValTiles is None:
# 			pValTiles = self.allTiles_probs_sorted
# 		
# 		fpValTiles = pValTiles ### If we need to modify the probability weight.
# 		modified_prob = fpValTiles/np.sum(fpValTiles)
# 		t_tiles = modified_prob * T_obs ### Time spent in each tile if not constrained
# 		t_tiles[t_tiles > 1200.0] = 1200.0 ### Upper limit of exposure time
# 		t_tiles[t_tiles < 60] = 60.0 ### Lower limit of exposure time
# 		Obs = np.cumsum(t_tiles) <= T_obs ### Tiles observable in T_obs seconds
# 		time_per_tile = t_tiles[Obs] ### Actual time spent per tile
# 		
# 		return time_per_tile
		



	def integrationTime(self, T_obs, pValTiles=None, func=None):
		'''
		METHOD :: This method accepts the probability values of the ranked tiles, the 
			  total observation time and the rank of the source tile. It returns 
			  the array of time to be spent in each tile which is determined based
			  on the localizaton probability of the tile. How the weight factor is 
			  computed can also be supplied in functional form. Default is linear.
				  
		pValTiles :: The probability value of the ranked tiles. Obtained from getRankedTiles 
					 output
		T_obs     :: Total observation time available for the follow-up.
		func	  :: functional form of the weight. Default is linear. 
					 For example, use x**2 to use a quadratic function.
		'''
		if pValTiles is None:
			pValTiles = self.allTiles_probs_sorted
		
		if func is None:
			f = lambda x: x
		else:
			f = lambda x: eval(func)
		fpValTiles = f(pValTiles)
		modified_prob = fpValTiles/np.sum(fpValTiles)
		t_tiles = modified_prob * T_obs ### Time spent in each tile if not constrained
		#t_tiles[t_tiles > 1200.0] = 1200.0 ### Upper limit of exposure time
		#t_tiles[t_tiles < 60] = 60.0 ### Lower limit of exposure time
		Obs = np.cumsum(t_tiles) <= T_obs ### Tiles observable in T_obs seconds
		time_per_tile = t_tiles[Obs] ### Actual time spent per tile
		
		return time_per_tile
		

	def optimize_time(self, T, M, range, pValTiles=None):
		'''
		METHOD	:: This method accepts the total duration of time, the absolute mag,
				   range for optimization(e.g: [0.0, 0.1]) and the probability values
				   of the tiles and returns an optimized array of time per tile.
		'''
		time_data, limmag_data, _ = np.loadtxt('timeMagnitude_new.dat', unpack=True)
		s = interpolate.UnivariateSpline(np.log(time_data), limmag_data, k=5)
		AA = np.linspace(range[0], range[1], 10000) ### Variable for optimization
		if pValTiles is None:
			pValTiles = self.allTiles_probs_sorted
		kappa_sum = []		
		for aa in AA:
# 			time_per_tile = self.integrationTime(T, pValTiles, func='x + ' + str(aa))
			time_per_tile = self.integrationTime(T, pValTiles, func='x**' + str(aa))
			limmag = s(np.log(time_per_tile))
			dists = 10**(1.0 + (limmag - M)/5)
# 			kappas = (dists**3)*pValTiles[:len(dists)]
			kappas = limmag*pValTiles[:len(dists)]
			kappa_sum.append(np.sum(kappas))
		kappa_sum = np.array(kappa_sum)
		maxIndex = np.argmax(kappa_sum)
		a_max = AA[maxIndex]
		kappa_max = kappa_sum[maxIndex]
		time_per_tile_max = self.integrationTime(T, pValTiles, func='x**' + str(a_max))
		return [time_per_tile_max, a_max]
		
############ UNDER CONSTRUCTION ############
	
class Scheduler(RankedTileGenerator):
	'''
	The scheduler class: Inherits from the RankedTileGenerator class. If no attribute 
	is supplied while creating schedular objects, a default instance of ZTF scheduler 
	is created. To generate scheduler for other telescopes use the corresponding site
	names which can be obtaine from astropy.coordinates.EarthLocation.get_site_names().
	The tile tile coordinate file also needs to be supplied to the variable tileCoord.
	This file needs to have at least three columns, the first being an ID (1, 2, ...),
	the second should be the tile center's ra value and the third the dec value of the 
	same. The utcoffset is the time difference between UTC and the site in hours. 
	'''
	def __init__(self, skymapFile, site='Palomar', 
				 tileCoord='ZTF_tiles_set1_nowrap_indexed.dat', utcoffset = -7.0):

		self.Observatory = EarthLocation.of_site(site)
		self.tileData = np.recfromtxt(tileCoord, names=True)
		self.skymapfile = skymapFile
		
		tileObj = RankedTileGenerator(skymapFile)
		[self.tileIndices, self.tileProbs] = tileObj.getRankedTiles()

		self.tiles = SkyCoord(ra = self.tileData['ra_center'][self.tileIndices]*u.degree, 
					    dec = self.tileData['dec_center'][self.tileIndices]*u.degree, 
					    frame = 'icrs') ### Tile(s)
		self.utcoffset = utcoffset*u.hour


	def tileVisibility(self, t, gps=False):
		'''
		METHOD	:: This method takes as input the time (gps or mjd) of observation
				   and the observatory site name, and returns the alt and az of the 
				   ranked tiles. It also returns the alt and az of the sun.
		t	    :: The time at which observation is made. Default is mjd. If time is 
				   given in gps then set gps to True.

		'''
		if gps: time = Time(t, format='gps') ### If time is given in GPS format
		else: time = Time(t, format='mjd') ### else time is assumed in mjd format
		altAz_tile = self.tiles.transform_to(AltAz(obstime=time, location=self.Observatory))
		altAz_sun = get_sun(time).transform_to(AltAz(obstime=time, location=self.Observatory))
		
		isSunDown = altAz_sun.alt.value < -18.0 ### Checks if it is past twilight.
		whichTilesUp = altAz_tile.alt.value > 20.0  ### Checks which tiles are up		
		
# 		return [altAz_tile, self.tileProbs, altAz_sun]
		return [self.tileIndices[whichTilesUp], self.tileProbs[whichTilesUp], altAz_sun]
		

	def advanceToSunset(self, eventTime, intTime):
		'''
		This method is called when the observation scheduler determines that the sun is 
		above horizon. It finds the nearest time prior to the next sunset within +/- 
		integration time and then advances the scheduler code to that point. 
		This speeds up the code by refraning from computing pointings during the daytime.
		Currently only works with GPS time. In the future mjd will also be included.
		
		eventTime	:: The GPS time for which the advancement is to be computed.
		intTim		:: The integration time for the obsevation
		'''
		
		dt = np.arange(0, 24*3600 + intTime, intTime)
		time = Time(eventTime + dt, format='gps')
		altAz_sun = get_sun(time).transform_to(AltAz(obstime=time, location=self.Observatory))
		timeBeforeSunset = (eventTime + dt)[altAz_sun.alt.value < -18.0][0] - intTime
		return timeBeforeSunset


	############### NOT TESTED ############
	def whenThisTileSets(self, index, currentTime, duration, gps=False):
		'''
		This method approximately computes the amount of time left in seconds for a tile
		to set below 20 degrees.
		
		index		::	The index of the tile for which setting tile is to be found
		currentTime	::	The current time when this tile is scheduled
		'''
# 		if gps: time = Time(currentTime, format='gps')
# 		else: time = Time(currentTime, format='mjd')
		thisTile = SkyCoord(ra = self.tileData['ra_center'][index]*u.degree, 
					    dec = self.tileData['dec_center'][index]*u.degree, 
					    frame = 'icrs') ### Tile(s)
		dt = np.arange(0, duration + 1.0, 1.0)
		times = Time(currentTime + dt, format='gps')
		altAz_tile = thisTile.transform_to(AltAz(obstime=times,
												location=self.Observatory))
		
		setTime = None
		if altAz_tile.alt.value[-1] < 20.0:
			s = interpolate.UnivariateSpline(altAz_tile.alt.value, times.value, k=3)
			setTime = s(20.0)
			
		return setTime

	def observationSchedule(self, duration, eventTime, integrationTime=120,
							observedTiles=None, plot=False, verbose=False):
		'''
		METHOD	:: This method takes the duration of observation, time of the GW trigger
				   integration time per tile as input and outputs the observation
				   schedule.
				   
		duration   		 :: Total duration of the observation in seconds.
		eventTime  		 :: The gps time of the time of the GW trigger.
		integrationTime  :: Time spent per tile in seconds (default == 120 seconds)
		observedTiles	 :: (Future development) Array of tile indices that has been 
							observed in an earlier epoch
		plot			 :: (optional) Plots the tile centers that are observed.
		verbose			 :: Toggle verbose flag for print statements.
				   
		
		'''
		
		includeTiles = np.cumsum(self.tileProbs) < 0.99
		includeTiles[np.sum(includeTiles)] = True
		
		thresholdTileProb = self.tileProbs[includeTiles][-1]



		observedTime = 0 ## Initiating the observed times
		elapsedTime = 0  ## Initiating the elapsed times. Time since observation begun.
		scheduled = np.array([]) ## tile indices scheduled for observation
		ObsTimes = []
		pVal_observed = []
		ii = 0
		observed_count = 0
		sun_ra = []
		sun_dec = []
		moon_ra = []
		moon_dec = []
		venus_ra = []
		venus_dec = []
		lunar_ilumination = []
		
		
		
		[_, _, altAz_sun] = self.tileVisibility(eventTime, gps=True)
		
		if altAz_sun.alt.value >= -18.0:
			if verbose: 
				localTime = Time(eventTime, format='gps') + self.utcoffset
				print(str(localTime.utc.datetime) + ': Sun above the horizon')
			eventTime = self.advanceToSunset(eventTime, integrationTime)
			if verbose:
				localTime = Time(eventTime, format='gps') + self.utcoffset
				print('Advancing time to ' + str(localTime.utc.datetime))
				print('\n')

		
		while elapsedTime <= duration: 
			[tileIndices, tileProbs, altAz_sun] = self.tileVisibility(eventTime, gps=True)
			localTime = Time(eventTime, format='gps') + self.utcoffset
			
			if altAz_sun.alt.value < -18.0: 
				if verbose: 
					print(str(localTime.utc.datetime) + ': Observation mode')
				for jj in np.arange(len(tileIndices)):
					if tileIndices[jj] not in scheduled:
						if tileProbs[jj] > thresholdTileProb:
							scheduled = np.append(scheduled, tileIndices[jj])
							ObsTimes.append(localTime)
							pVal_observed.append(tileProbs[jj])
							Sun = get_sun(Time(eventTime, format='gps'))
							sun_ra.append(Sun.ra.value)
							sun_dec.append(Sun.dec.value)
							Moon = get_moon(Time(eventTime, format='gps'))
							sunMoonAngle = Sun.separation(Moon)
							phaseAngle = np.arctan2(Sun.distance*np.sin(sunMoonAngle), 
										Moon.distance - Sun.distance *
										np.cos(sunMoonAngle))
							illumination = 0.5*(1.0 + np.cos(phaseAngle))
							
							if verbose: print('Lunar illumination = ' + str(illumination))
							lunar_ilumination.append(illumination)
							
							moon_ra.append(Moon.ra.value)
							moon_dec.append(Moon.dec.value)
							observedTime += integrationTime ## Tracking observations
							break
				
			else:
				if verbose: 
					localTime = Time(eventTime, format='gps') + self.utcoffset
					print(str(localTime.utc.datetime) + ': Sun above the horizon')
				eventTime = self.advanceToSunset(eventTime, integrationTime)
				if verbose:
					localTime = Time(eventTime, format='gps') + self.utcoffset
					print('Advancing time to ' + str(localTime.utc.datetime))
					print('\n')
			

			ii += 1

			eventTime += integrationTime
			elapsedTime += integrationTime
			print('elapsedTime --->' + str(elapsedTime))
			print('observedTime --->' + str(observedTime))


	
# 		while observedTime <= duration:
# 		while elapsedTime <= duration: 
# 			print('Observed time = ' + str(observedTime/3600.))
# 			dt = integrationTime * ii
# 			elapsedTime += integrationTime
# 			print('**** Elapsed time = ' + str(elapsedTime) + '****')
# 			[tileIndices, tileProbs, altAz_sun] = self.tileVisibility(eventTime,
# 																		 gps=True)
# 			localTime = Time(eventTime, format='gps') + self.utcoffset
# 			
# 
# 			
# 			if altAz_sun.alt.value < -18.0: ### Sun below horizon
# 				observed_count += 1
# 				observedTime += integrationTime ### Augment observed time
# 				if verbose: 
# 					print(str(localTime.utc.datetime) + ': Observation mode')
# 
# 				for jj in np.arange(len(tileIndices)):
# 					if tileIndices[jj] not in scheduled:
# 						if tileProbs[jj] > thresholdTileProb:
# 							scheduled = np.append(scheduled, tileIndices[jj])
# 							ObsTimes.append(localTime)
# 							pVal_observed.append(tileProbs[jj])
# 							Sun = get_sun(Time(eventTime, format='gps'))
# 							sun_ra.append(Sun.ra.value)
# 							sun_dec.append(Sun.dec.value)
# 							Moon = get_moon(Time(eventTime, format='gps'))
# 							sunMoonAngle = Sun.separation(Moon)
# 							phaseAngle = np.arctan2(Sun.distance*np.sin(sunMoonAngle), 
# 										Moon.distance - Sun.distance *
# 										np.cos(sunMoonAngle))
# 							illumination = 0.5*(1.0 + np.cos(phaseAngle))
# 							
# 							if verbose: print('Lunar illumination = ' + str(illumination))
# 							lunar_ilumination.append(illumination)
# 							
# 							moon_ra.append(Moon.ra.value)
# 							moon_dec.append(Moon.dec.value)
# 							break
# 
# 			else:
# 				if verbose: 
# 					localTime = Time(eventTime, format='gps') + self.utcoffset
# 					print(str(localTime.utc.datetime) + ': Sun above the horizon')
# 				eventTime = self.advanceToSunset(eventTime, integrationTime)
# 				if verbose:
# 					localTime = Time(eventTime, format='gps') + self.utcoffset
# 					print('Advancing time to ' + str(localTime.utc.datetime))
# 					print('\n')




		for ii in np.arange(len(scheduled)):
			print(str(ObsTimes[ii].utc.datetime) + '\t' + str(int(scheduled[ii])))
			
		pVal_observed = np.array(pVal_observed)
		sun_ra = np.array(sun_ra)
		sun_dec = np.array(sun_dec)
		moon_ra = np.array(moon_ra)
		moon_dec = np.array(moon_dec)
		venus_ra = np.array(venus_ra)
		venus_dec = np.array(venus_dec)
		
		return [scheduled.astype('int'), pVal_observed, sun_ra, 
				sun_dec, moon_ra, moon_dec, lunar_ilumination]












####################END OF CLASS METHODS########################


def evolve_abs_Mag(dt, model, offset=0):	### UNDERDEVELOPMENT ###
	'''
	METHOD	:: This method takes as input the light curve model and the time
			   since the merger and outputs the absolute magnitude of the 
			   source.
			   
	dt 	 	:: Time since merger
	model	:: The light curve model. Right now only one model (NSNS_MNmodel1_FRDM_r)
	offset	:: (Optional) The offset of the peak of the light curve from the merger.
	'''

	data = np.recfromtxt(model, names=True)
	s = interpolate.UnivariateSpline(data['time'], data['magnitude'], k=5)
	mag = s(dt - offset)

	
def gaussian_distribution_function(x, mu, sigma):
		'''
		METHOD	:: Creates the gaussian function corresponding to the
				   mean and standard deviation of the limiting magnitude
				   corresponding to the given time
		'''
		return 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))

def apparent_from_absolute_mag(absolute_mag, source_dist_parsec):
		'''
		METHOD	:: This method computes the aparent magnitude from the absolute
		magnitude model and the distance to the source.
		'''
		return absolute_mag + 5*np.log10(source_dist_parsec/10.)

def detectability(rank, time_per_tile, total_observation_time, absolute_mag, source_dist_parsec, time_data, limmag_data, error_data = None, verbose=False):
	'''
	METHOD :: This method takes as input the time allotted per tile, 
	total observation time allotted for an event, the absolute 
	magnitude model of the source and the dependence of 
	limiting magnitude on integration time, and returns either a 
	boolean numpy object[True/False] whether the source can be detected, or
	a probability of detection if the error data (sigma) for 
	the limiting magnitude is provided.
		  
	total_observation_time 	 :: The total observation time for the event
	absolute_magnitude 		 :: The absolute magnitude of the source that 
					is to be set by the model.
	source_dist_parsec 		 :: distance to source in parsecs 
	time_data, 
	limmag_data, 
	error_data 				 :: The data which needs to be interpolated to give 
								limiting magnitude as a function of time. error_data, 
								if provided, will allow for a detection probability 
								to be generated as output. If detectability(rank,
								time_per_tile, total_observation_time, absolute_mag,
								source_dist_parsec, time_data, limmag_data, 
								error_data = None, verbose=False)
								error_data is not provided, a Boolean (True/False) for
								detection will be output. 
	'''
	### Convert to numpy object if scalar supplied
	if isinstance(time_per_tile, (np.ndarray,)) == False:
		time_per_tile = np.array(time_per_tile)
	### Check if source tile rank has been reached. Return non-detection if not
	rank_reached		= (total_observation_time/time_per_tile).astype(int)
	rank_reached_mask	= rank_reached > rank	# True means rank is reachable
	if np.all(~rank_reached_mask):	# if rank cannot be reached for 
									# any integration time	
		if verbose: print("Tile not reached in ANY allotted observation time")
		if error_data is not None:
			return np.zeros(len(time_per_tile))
		else:
			return rank_reached_mask

	### Determine limiting magnitude as a function of time, via interpolation of data
	s = interpolate.UnivariateSpline(np.log(time_data), limmag_data, k=5)
	limmag = s(np.log(time_per_tile))
	apparent_mag = apparent_from_absolute_mag(absolute_mag, source_dist_parsec)
	### If error_data is not supplied, return Boolean True/False for detection
	if error_data is None:
		depthReached = (limmag > apparent_mag)
		if np.any(depthReached) is False:
			if verbose: print("Depth not reached in ANY allotted integration time")
		return np.logical_and(depthReached, rank_reached_mask)
		### Both Depth criteria and rank criteria should be satisfied
	### If error_data is supplied, return detection probability
	else:
		s_err = interpolate.UnivariateSpline(np.log(time_data), error_data, k=5)
		mu = limmag
		sigma = s_err(np.log(time_per_tile))
		very_large_number = 1000 #proxy for +infinity
		samples = 10**5
		x = np.linspace(apparent_mag, very_large_number, samples, endpoint = True)
		### If floats are passed to the function, return the answer rightaway
		if isinstance(mu, (float, np.float, np.float64,)) == True and isinstance(sigma, (float, np.float, np.float64,)) == True:
			y = gaussian_distribution_function(x, mu, sigma)
			return np.trapz(y,x)
		# If an array of time_per_tile had been passed
		# Note that mu and sigma are equal length arrays, as defined above
		else:
			result = []
			for ii in range(len(mu)):
				y = gaussian_distribution_function(x,mu[ii],sigma[ii])
				result.append(np.trapz(y,x))
			return np.array(result)
