# Copyright (C) 2018 David Kaplan
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
from astropy.table import Table
from astropy import constants as c, units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy
import healpy
# import cPickle as pickle
import pickle

def inside(X,Y,x0,y0):
    """
    result=inside(X,Y,x0,y0)
    see if the points (x0,y0) is inside the convex polygon defined by (X,Y)
    X,Y and x0,y0 should be 1D arrays
    based on:
    http://demonstrations.wolfram.com/AnEfficientTestForAPointToBeInAConvexPolygon/
    """
    X=X-np.array([x0]).T
    # account for wraps in RA
    if (X>180).any():
        X[X>=180]-=360
    if (X<-180).any():
        X[X<=-180]+=360
    # we won't be able to deal with wraps in Dec
    # so we don't want to do this for very close to the poles
    Y=Y-np.array([y0]).T
    n=X.shape[1]
    aplus=np.ones((X.shape[0]),dtype=np.bool)
    aminus=np.ones((X.shape[0]),dtype=np.bool)
    for i1 in range(n):
        i2=(i1+1) % n
        a=X[:,i2]*Y[:,i1]-X[:,i1]*Y[:,i2]
        aplus=aplus & (a>0)
        aminus=aplus & (a<0)
    return aplus | aminus

def order(X,Y,x0,y0):
    """
    result=order(X,Y,x0,y0)

    X,Y and x0,y0 should be 1D arrays
    based on:
    http://demonstrations.wolfram.com/AnEfficientTestForAPointToBeInAConvexPolygon/
    """
    X=X-x0
    # account for wraps in RA
    if (X>180).any():
        X[X>=180]-=360
    if (X<-180).any():
        X[X<=-180]+=360
    # we won't be able to deal with wraps in Dec
    # so we don't want to do this for very close to the poles
    Y=Y-y0
    angles=np.arctan2(Y,X)
    return np.argsort(angles)

class ZTFtile():

    def __init__(self, RA, Dec, config_struct, number=None):
        """
        z=ZTFtile(RA, Dec, number)
        creates a ZTFtile object centered at the given coordinates
        coordinates can be Quantities (with units) or floats (assumed degrees)

        """
        quadrantsFile = config_struct["quadrantsFile"]

        t=Table.read(quadrantsFile,format='ascii.commented_header')
        # Cartesian coordinates in the plane of projection
        self.quadrant_x=t['DX']*u.deg
        self.quadrant_y=t['DY']*u.deg
        self.quadrant_size=np.array([3072,3080])
        self.quadrant_scale=1*u.arcsec

        if isinstance(RA, astropy.units.quantity.Quantity):            
            self.RA=RA
        else:
            self.RA=RA*u.deg
        if isinstance(Dec, astropy.units.quantity.Quantity):            
            self.Dec=Dec
        else:
            self.Dec=Dec*u.deg

        self.number=number

        self.quadrant_RA,self.quadrant_Dec=self.quadrant_centers()
        
        self._wcs=[]
        for i in range(len(self.quadrant_RA)):
            self._wcs.append(WCS(naxis=2))
            self._wcs[-1].wcs.crpix=(self.quadrant_size+1.0)/2
            self._wcs[-1].wcs.ctype=['RA---TAN','DEC--TAN']
            self._wcs[-1].wcs.crval=[self.quadrant_RA[i].value,
                                     max(min(self.quadrant_Dec[i].value,90),-90)]
            self._wcs[-1].wcs.cd=[[-self.quadrant_scale.to(u.deg).value,0],
                                  [0,-self.quadrant_scale.to(u.deg).value]]
    def quadrant_centers(self):
        """
        alpha,delta=quadrant_centers()
        return celestial coordinates for the centers of each quadrant
        given the pointing center RA,Dec
        """
        # convert to Native longitude (phi) and latitude (theta)
        # need intermediate step (Rtheta) to deal with TAN projection
        # Calabretta & Greisen (2002), Eqn. 14,15
        phi=np.arctan2(self.quadrant_x,-self.quadrant_y)
        Rtheta=np.sqrt(self.quadrant_x**2+self.quadrant_y**2)
        # now to theta using the TAN projection
        # Calabrett & Greisen (2002), Eqn. 55
        theta=np.arctan2(1,Rtheta.to(u.rad).value)*u.rad
        
        # central point of projection (native pole)
        alphap=self.RA
        deltap=self.Dec
        # Native longitude/latitude of fiducial point
        # appropriate for zenithal projections including TAN
        phi0=0*u.deg
        theta0=90*u.deg
        # Native longitude/latitue of celestial pole
        # for delta0<theta0 then phip should be 180 deg
        # and theta0=90 deg
        phip=180*u.deg
        thetap=self.Dec
        
        # Celestial longitude/latitude
        # Calabretta & Greisen (2002), Eqn. 2
        alpha=alphap+np.arctan2(-np.cos(theta)*np.sin(phi-phip),
                                np.sin(theta)*np.cos(deltap)-np.cos(theta)*np.sin(deltap)*np.cos(phi-phip))
        
        delta=(np.arcsin(np.sin(theta)*np.sin(deltap)+np.cos(theta)*np.cos(deltap)*np.cos(phi-phip))).to(u.deg)
        alpha[alpha<0*u.deg]+=360*u.deg
        return alpha,delta

    def _inside(self, RA, Dec):
        """
        _inside(RA,Dec)
        returns whether a given RA,Dec (floats in degrees) is inside the FOV
        """
        on=np.zeros_like(RA)
        for i in range(len(self._wcs)):
            footprint=self._wcs[i].calc_footprint(axes=self.quadrant_size)
            on+=inside(footprint[:,0],footprint[:,1],RA,Dec)
            
        return (on>0)

    def _inside_nogaps(self, RA, Dec):
        """
        _inside_nogaps(RA,Dec)
        returns whether a given RA,Dec (floats in degrees) is inside the rough FOV
        ignores chipgaps
        """
        # get all of the corners of the quadrants
        footprint=np.zeros((4*len(self._wcs),2))
        for i in range(len(self._wcs)):
            footprint[(4*i):(4*(i+1)),:]=self._wcs[i].calc_footprint(axes=self.quadrant_size)
        corners=SkyCoord(footprint[:,0]*u.deg,footprint[:,1]*u.deg)
        # distances from these corners to the central pointing position
        corner_distances=corners.separation(SkyCoord(self.RA,self.Dec))
        # find the 4 furthest points - these will define the outer boundaries
        outer_corners=footprint[np.argsort(corner_distances)[-4:],:]
        # sort them by angle to make sure they are convex
        corner_order=order(outer_corners[:,0],outer_corners[:,1],
                           self.RA.value,self.Dec.value)[::-1]
        outer_corners=outer_corners[corner_order,:]
        on=inside(outer_corners[:,0],outer_corners[:,1],RA,Dec)
        return (on>0)

    def inside(self, *args):
        """
        inside(*args)
        returns whether a given coordinate is inside the FOV
        coordinate can be SkyCoord
        or separate RA,Dec
        RA,Dec can be Quantity (with units) or floats (in degrees)
        """

        if len(args)==1:
            if isinstance(args[0],astropy.coordinates.sky_coordinate.SkyCoord):
                return self._inside(args[0].ra.value,args[0].dec.value)
        elif len(args)==2:
            if isinstance(args[0],astropy.units.quantity.Quantity):
                return self._inside(args[0].value,args[1].value)
            else:
                return self._inside(args[0],args[1])
            
    def inside_nogaps(self, *args):
        """
        inside_nogaps(*args)
        returns whether a given coordinate is inside the rough FOV
        ignoring the chip gaps
        coordinate can be SkyCoord
        or separate RA,Dec
        RA,Dec can be Quantity (with units) or floats (in degrees)
        """

        if len(args)==1:
            if isinstance(args[0],astropy.coordinates.sky_coordinate.SkyCoord):
                return self._inside(args[0].ra.value,args[0].dec.value)
        elif len(args)==2:
            if isinstance(args[0],astropy.units.quantity.Quantity):
                return self._inside_nogaps(args[0].value,args[1].value)
            else:
                return self._inside_nogaps(args[0],args[1])

class HP_coverage:

    def __init__(self, hpfile, nside=256, verbose=False):
        """
        HP_coverage(hpfile, nside=256, verbose=False)
        healpix coverage map
        input healpix file and desired nside for degradation (if needed)
        """

        self.nside=nside
        self.npix=healpy.nside2npix(self.nside)

        # read the healpix map
        if verbose:
            print('Loading %s' % hpfile)
        b,h=healpy.read_map(hpfile,h=True,nest=False,verbose=verbose)
        # convert the header to a dictionary
        h={x[0]: x[1] for x in h}
        self.nside_in=h['NSIDE']
        self.npix_in=healpy.nside2npix(self.nside_in)
        if verbose:
            print('Read %s with NSIDE=%d' % (hpfile, self.nside_in))

        # degrade if necessary
        if self.nside < self.nside_in:
            if verbose:
                print('Converting to NSIDE=%d' % self.nside)
            self.hpmap=healpy.ud_grade(b, self.nside, power=-2)
        else:
            self.hpmap=b
            self.nside=self.nside_in
            self.npix=healpy.nside2npix(self.nside)

        # coordinates for each pixel in the map
        RA,Dec=healpy.pix2ang(self.nside, np.arange(self.npix),lonlat=True)
        self.RA=RA*u.deg
        self.Dec=Dec*u.deg

    def tile_coverage(self, ztftiles):
        """
        covered=tile_coverage(ztftiles)
        returns an array the same size as the healpix map
        composed of booleans to show which pixels are covered
        ztftiles is list of ZTFtile objects()
        """

        try:
            l=len(ztftiles)
        except:
            ztftiles=[ztftiles]

        covered=np.zeros(len(self.hpmap),dtype=np.bool)
        for itile in range(len(ztftiles)):
            covered[ztftiles[itile].inside(self.RA,self.Dec)]=True
        return covered
    
    def tile_probability(self, ztftiles):
        """
        probability=tile_probability(tile_RA,tile_Dec)
        ztftiles is list of ZTFtile objects()
        returns total covered probability
        """

        covered=self.tile_coverage(ztftiles)
        return (self.hpmap*covered).sum()

    def get_tile_values(self, hpmap, ztftiles):
        """
        values=get_tile_values(hpmap, ztftiles)
        gets the values used for sorting the tile contributions
        this is implemented as the integral of the probability for the given map over each tile
        ztftiles is list of ZTFtile objects()
        """
        
        values=np.zeros(len(ztftiles))
        for itile in range(len(ztftiles)):
            values[itile]=hpmap[ztftiles[itile].inside(self.RA,self.Dec)].sum()
        return values
    
    def find_tiles(self, ztftiles, probability_target=0.9, verbose=False):
        """
        tiles,probability=find_tiles(ztftiles, probability_target=0.9, verbose=False)

        give ztftiles (list of ZTFtile objects())

        will find the combination to achieve the total probability target
        """

        # starting values and order
        tile_values=self.get_tile_values(self.hpmap, ztftiles)
        tile_order=np.argsort(tile_values)[::-1]

        tiles=[]
        # this is a copy of the map that we can modify
        hpmapc=np.copy(self.hpmap)
        covered=np.zeros(len(self.hpmap),dtype=np.bool)
        summed_probability=0
        while summed_probability < probability_target and len(tiles)<len(ztftiles):
            # add it to the list
            tiles.append(tile_order[0])

            individual_prob=(hpmapc[ztftiles[tile_order[0]].inside(self.RA,self.Dec)].sum())
            if verbose:
                print('Adding tile %d (%f,%f): individual probability = %.3f, total probability = %.3f' % (tile_order[0],
                                                                                                           ztftiles[tile_order[0]].RA.value,
                                                                                                           ztftiles[tile_order[0]].Dec.value,
                                                                                                           individual_prob,
                                                                                                           summed_probability+individual_prob))
            covered[ztftiles[tile_order[0]].inside(self.RA,self.Dec)]=True
            summed_probability+=individual_prob
            hpmapc[ztftiles[tile_order[0]].inside(self.RA,self.Dec)]=0
                
            # redo the priorities to account for the new 
            # probability values
            tile_values=self.get_tile_values(hpmapc, 
                                             ztftiles)
            # priorities for each of those
            tile_order=np.argsort(tile_values)[::-1]

        return tiles,summed_probability

    def percentile(self, level):
        """
        value=percentile(level)
        determines the value of the map above which there is level of total probabiilty
        """
        m=np.sort(self.hpmap)[::-1]
        msum=np.cumsum(m)
        include=msum<level*m.sum()
        # add one more point just to be sure
        if include.sum()<len(m):
            include[include.sum()]=True
        return m[include][-1]

    def percentile_area(self, level):
        """
        area=percentile_area(level)
        determines the area of the map above which there is level of total probabiilty        
        """
        return (self.hpmap>=self.percentile(level)).sum()*healpy.nside2pixarea(self.nside,degrees=True)*u.deg**2
