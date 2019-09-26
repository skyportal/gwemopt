# Copyright (C) 2019 Siddharth Mohite
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

from __future__ import print_function
import numpy as np
import healpy as hp
from astropy.table import Table
from astropy import constants as c, units as u
from astropy.coordinates import SkyCoord, ICRS, FK5
from astropy.wcs import WCS
import astropy
import healpy
# import cPickle as pickle
import pickle
import os

import calculate_ccd_coords as calc_ccd


class DECamtile:
    def __init__(self, RA, Dec, number=None, missing=None):
        """
        z=DECamtile(RA, Dec, number, missing)
        creates a DECamtile object centered at the given coordinates
        coordinates can be Quantities (with units) or floats (assumed degrees)
        number is just used for informational purposes (bookkeeping)
        missing lists the missing ccds (potentially None or [])
        if missing is not None, should be a list of ccds (1-62)
        they will be removed from the ccd-by-ccd WCS

        """
        self.ccd_size=np.array([2046,4094])
#        self.quadrant_scale=0.27*u.arcsec

        if isinstance(RA, astropy.units.quantity.Quantity):            
            self.RA=RA
        else:
            self.RA=RA*u.deg
        if isinstance(Dec, astropy.units.quantity.Quantity):            
            self.Dec=Dec
        else:
            self.Dec=Dec*u.deg

        self.number=number

        centers = self.ccd_centers()
        self.ccd_RA=np.array([centers[key][0] for key in centers.keys()])*u.deg
        self.ccd_Dec=np.array([centers[key][1] for key in centers.keys()])*u.deg
        cd = calc_ccd.get_cd(filename='c4d_140819_032452_ooi_g_v1.fits.fz') 
        
        self.missing_ccds=[]
        if missing is not None and len(missing)>0:
            for m in missing:
                if isinstance(m,int) or len(m)==1:
                    # assume this is 0 origin
                    self.missing_ccds.append(m)
#                elif len(m)==2:
                    # this assumes CCD and quadrant numbers are 1-origin
#                    self.missing_quadrants.append(4*(m[0]-1)+(m[1]-1))
        good_ccds=np.setdiff1d(np.arange(len(self.ccd_RA)),
                                    np.array(self.missing_ccds))
        
        self._wcs=[]
        for i in good_ccds:
            self._wcs.append(self.ccd_WCS(cd,self.ccd_RA[i],self.ccd_Dec[i]))

    def ccd_WCS(self, cd, ccd_RA, ccd_Dec):
        """
        w=ccd_WCS(cd,crpix,ccd_RA, ccd_Dec)
        returns a WCS object that is specific to the ccd RA and Dec specified
        overall scale, size are determined by class variables
        """
        
        w=WCS(naxis=2)
        w.wcs.crpix=(self.ccd_size+1.0)/2
        w.wcs.ctype=['RA---TAN','DEC--TAN']
        w.wcs.crval=[ccd_RA.value,
                     max(min(ccd_Dec.value,90),-90)]
        w.wcs.cd=cd
        return w

    def ccd_centers(self):
        """
        alpha,delta=ccd_centers()
        return celestial coordinates for the centers of each ccd
        given the pointing center RA,Dec
        """
        ccd_centers_radec = calc_ccd.ccd_xy_to_radec(self.RA,self.Dec,self.get_ccd_centers())
        return ccd_centers_radec

    def get_ccd_centers(self):
        ccd_cent_xy = calc_ccd.get_ccdcenters_xy('c4d_140819_032452_ooi_g_v1.fits.fz')
        return ccd_cent_xy

class CCDProb:
    '''
    Class :: Instantiate a CCDProb object that will allow us to calculate the
             probability content in a single CCD.
    '''
    def __init__(self, RA, Dec):
        self.RA = RA
        self.Dec = Dec
        self.ccd_size = np.array([2046, 4094])

def get_decam_ccds(ra,dec,save_footprint=False):
    """Calculate DECam CCD footprints as offsets from the telescope
    boresight. Also optionally saves the footprints in a region file 
    called footprint.reg"""
#    ccd_prob = CCDProb(ra, dec)
    decam_tile = DECamtile(ra, dec)
    ccd_cents_ra = decam_tile.ccd_RA
    ccd_cents_dec = decam_tile.ccd_Dec
    offsets = np.asarray([decam_tile._wcs[ccd_id].calc_footprint(axes=decam_tile.ccd_size)
            for ccd_id in range(len(ccd_cents_ra))])
    if save_footprint:
        ra,dec = np.transpose(offsets, (2, 0, 1))
        with open('footprint.reg','a') as f:
            for i in range(len(ra)): 
                lines = ['# Region file format: DS9 version 4.0 \n', '# global color=green font="helvetica 12 bold select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source \n', 'ICRS \n', 'polygon('+str(ra[i,0]) +',' + str(dec[i,0]) + ',' + str(ra[i,1]) +',' + str(dec[i,1]) + ',' + str(ra[i,2]) +',' + str(dec[i,2]) + ',' + str(ra[i,3]) +',' + str(dec[i,3]) + ') # color=green, width=2 \n'] 
                f.writelines(lines)
            f.close()
    return np.transpose(offsets, (2, 0, 1))

def get_quadrant_ipix(nside, ra, dec):

    ccd_coords = get_decam_ccds(ra,dec)

    skyoffset_frames = SkyCoord(ra, dec, unit=u.deg).skyoffset_frame()
    ccd_coords_icrs = SkyCoord(
                    *np.tile(
                        ccd_coords[:, np.newaxis, ...],
                        (1, 1, 1)), unit=u.deg,
                    frame=skyoffset_frames[:, np.newaxis, np.newaxis]
                ).transform_to(ICRS)
    ccd_xyz = np.moveaxis(
        ccd_coords_icrs.cartesian.xyz.value, 0, -1)[0]

    ipixs = []
    for subfield_id, xyz in enumerate(ccd_xyz):
        ipix = hp.query_polygon(nside, xyz)
        ipixs.append(ipix.tolist())
    return ipixs

