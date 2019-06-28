#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: David Corre

import numpy as np
from astropy.table import Table
from astropy import units as u
from math import log
from mocpy import MOC
import json
import healpy as hp

class MOC_confidence_region:
    """
    This class converts the probability skymap from the fits to a MOC in json format
        Input:
             path: string
                   path where skymap is located
            
             filename: string
                   LVC probability sky map name

             percentage: float
                      probability percentage of the enclosed area  
    """

    def __init__(self, path, filename, percentage):
        # Path to the skymap fits file (ending with '/')
        self.path = path
        # Name of the skymap fits file
        self.filename = filename
        # Desired cumulative probability of the skymap. Given in %
        if percentage > 1:
            self.percentage = float(percentage / 100)
        else:
            self.percentage = percentage


    def create_moc(self):
        """
        Multi-Order coverage map (MOC) of sky area enclosed within a contour plot
        at a given confidence level.
    
        (Function adapted from Giuseppe Greco github repository)
        """

        #reading skymap
        hpx = hp.read_map( self.path+self.filename, verbose = False )
        npix = len( hpx )
        nside = hp.npix2nside( npix )
 
        # Save nside
        self.nside = nside

        sort = sorted( hpx, reverse = True )
        cumsum = np.cumsum( sort )
        index, value = min( enumerate( cumsum ), key = lambda x: abs( x[1] - self.percentage ) )

        # finding ipix indices confined in a given percentage 
        index_hpx = range( 0, len( hpx ) )
        hpx_index = np.c_[ hpx, index_hpx ]

        sort_2array = sorted( hpx_index, key = lambda x: x[0], reverse = True )
        value_contour = sort_2array[ 0:index ]

        j = 1 
        table_ipix_contour = [ ]

        for i in range ( 0, len( value_contour ) ):
            ipix_contour = int( value_contour[i][j] )
            table_ipix_contour.append( ipix_contour )
          
        # from index to polar coordinates
        theta, phi = hp.pix2ang( nside, table_ipix_contour )

        # converting these to right ascension and declination in degrees
        ra = np.rad2deg( phi )
        dec = np.rad2deg( 0.5 * np.pi - theta )

        # Save ra, dec of the pixel associated with highest probability
        self.RA, self.DEC = ra[0], dec[0]

        # creating an astropy.table with RA[deg] and DEC[deg] ipix positions
        contour_ipix = Table([ ra, dec ], names = ('RA[deg]', 'DEC[deg]'), 
                             meta = {'ipix': 'ipix table'})

        # setting MOC order
        moc_order = int( log( nside, 2 ) )

        # creating a MOC map from the contour_ipix table
        #moc = MOC.from_table( contour_ipix, 'RA[deg]', 'DEC[deg]', moc_order )
        moc = MOC.from_lonlat(contour_ipix['RA[deg]'].T * u.deg, contour_ipix['DEC[deg]'].T * u.deg,moc_order)

        # writing MOC file in fits
        #moc.write(path=short_name + '_MOC_' + str( percentage ) +'.json',format='json')

        # Serialise to json dictionary
        self.moc = moc.serialize(format='json')
