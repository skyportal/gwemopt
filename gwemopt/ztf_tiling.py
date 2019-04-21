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
import healpy as hp
from astropy.table import Table
from astropy import constants as c, units as u
from astropy.coordinates import SkyCoord, ICRS
from astropy.wcs import WCS
import astropy
import healpy
# import cPickle as pickle
import pickle
import os


class ZTFtile:

    # make sure it can find the table of quadrant centers
    #classlocation=os.path.join(os.path.split(__file__)[0],'../tiling')
    #t=Table.read(os.path.join(classlocation,'ZTF_quadrantcenters.dat'),
    #             format='ascii.commented_header')
    # Cartesian coordinates in the plane of projection
    #quadrant_x=t['DX']*u.deg
    #quadrant_y=t['DY']*u.deg

    def __init__(self, RA, Dec, number=None, missing=None):
        """
        z=ZTFtile(RA, Dec, number, missing)
        creates a ZTFtile object centered at the given coordinates
        coordinates can be Quantities (with units) or floats (assumed degrees)
        number is just used for informational purposes (bookkeeping)
        missing lists the missing quadrants (potentially None or [])
        if missing is not None, should be a list of quadrants (0--63) or CCD,quadrant pairs (1-16,1-4)
        they will be removed from the quadrant-by-quadrant WCS

        """
        quad = self.get_quadrant_centers()
        self.quadrant_x=quad[:,2]*u.deg
        self.quadrant_y=quad[:,3]*u.deg
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
        
        self.missing_quadrants=[]
        if missing is not None and len(missing)>0:
            for m in missing:
                if isinstance(m,int) or len(m)==1:
                    # assume this is 0 origin
                    self.missing_quadrants.append(m)
                elif len(m)==2:
                    # this assumes CCD and quadrant numbers are 1-origin
                    self.missing_quadrants.append(4*(m[0]-1)+(m[1]-1))
        good_quadrants=np.setdiff1d(np.arange(len(self.quadrant_RA)),
                                    np.array(self.missing_quadrants))
        
        self._wcs=[]
        for i in good_quadrants:
            self._wcs.append(self.quadrant_WCS(self.quadrant_RA[i],self.quadrant_Dec[i]))

    def quadrant_WCS(self, quadrant_RA, quadrant_Dec):
        """
        w=quadrant_WCS(quadrant_RA, quadrant_Dec)
        returns a WCS object that is specific to the quadrant RA and Dec specified
        overall scale, size are determined by class variables
        """
        
        w=WCS(naxis=2)
        w.wcs.crpix=(self.quadrant_size+1.0)/2
        w.wcs.ctype=['RA---TAN','DEC--TAN']
        w.wcs.crval=[quadrant_RA.value,
                     max(min(quadrant_Dec.value,90),-90)]
        w.wcs.cd=[[-self.quadrant_scale.to(u.deg).value,0],
                  [0,-self.quadrant_scale.to(u.deg).value]]
        return w

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

    def get_quadrant_centers(self):
        quad = [[1, 1, 3.2210639743885596, -2.4437146770476383],
                [1, 2, 2.3535001501796575, -2.43881516574295],
                [1, 3, 2.3503417727072713, -3.3081079798782698],
                [1, 4, 3.2184995347493874, -3.3135486187697305],
                [2, 1, 1.360399735814718, -2.4401317085420295],
                [2, 2, 0.4945677283623324, -2.43770260896455],
                [2, 3, 0.49257586768271355, -3.3063833103517815],
                [2, 4, 1.3589823960224123, -3.308995989160921],
                [3, 1, -0.4933901562131375, -2.434813777659866],
                [3, 2, -1.359168612669584, -2.4315618710281117],
                [3, 3, -1.3633930406282657, -3.3004078547524585],
                [3, 4, -0.4970843882478723, -3.3035234264684044],
                [4, 1, -2.3550841263512168, -2.4300020287485142],
                [4, 2, -3.2225696141613875, -2.4288711155723512],
                [4, 3, -3.225987817653737, -3.2986410207627994],
                [4, 4, -2.3578968641038385, -3.29921741067662],
                [5, 1, 3.2270472994872974, -0.5260446790890496],
                [5, 2, 2.360048851811889, -0.5236056946633051],
                [5, 3, 2.3580069857424673, -1.391013438779468],
                [5, 4, 3.2252436129820534, -1.3940385615739272],
                [6, 1, 1.3636942887667292, -0.5244366494318076],
                [6, 2, 0.4985661268688817, -0.5224684314946721],
                [6, 3, 0.4966475798958921, -1.3892521474803983],
                [6, 4, 1.3619833350672432, -1.3914104228161555],
                [7, 1, -0.4885352259899877, -0.5165091052597491],
                [7, 2, -1.353711077416428, -0.5152297582308262],
                [7, 3, -1.3552233617363139, -1.3821691302104016],
                [7, 4, -0.489935604981924, -1.3833014709475784],
                [8, 1, -2.35112039116874, -0.5124206296644447],
                [8, 2, -3.2179755018158858, -0.5120272575149895],
                [8, 3, -3.218910321057886, -1.3798304494175564],
                [8, 4, -2.3518957475858957, -1.3797479763456497],
                [9, 1, 3.2315748392794528, 1.3889779258968369],
                [9, 2, 2.364458464672072, 1.3900361847270215],
                [9, 3, 2.362357247822555, 0.5226598175841949],
                [9, 4, 3.229340886555462, 0.5210066079569343],
                [10, 1, 1.3683735895823672, 1.39427691039317],
                [10, 2, 0.5029965829034689, 1.3958587387612276],
                [10, 3, 0.5011665116661931, 0.5290203277616491],
                [10, 4, 1.366338951342555, 0.5272736507052246],
                [11, 1, -0.4822088351788257, 1.399285270694412],
                [11, 2, -1.347565895275316, 1.4010754421142027],
                [11, 3, -1.3489171520377228, 0.5341040741152838],
                [11, 4, -0.4837320077661326, 0.5324788914730108],
                [12, 1, -2.3424336622973194, 1.4040639060045907],
                [12, 2, -3.2095684182672755, 1.4052262304759773],
                [12, 3, -3.2095018762120544, 0.5372847732203613],
                [12, 4, -2.342590443769107, 0.5366672655753082],
                [13, 1, 3.2399153631397453, 3.309526119435319],
                [13, 2, 2.3717429700729746, 3.3104062851928253],
                [13, 3, 2.3685575187354933, 2.441059229055063],
                [13, 4, 3.2361524599183253, 2.4396874072033157],
                [14, 1, 1.3776907334001645, 3.309296478848194],
                [14, 2, 0.511265387229377, 3.3129964364715527],
                [14, 3, 0.507007945159137, 2.444240399708313],
                [14, 4, 1.372808010956761, 2.440396042373973],
                [15, 1, -0.47857144910010013, 3.31541767888218],
                [15, 2, -1.3449638389147003, 3.3156973843720072],
                [15, 3, -1.3440992664923928, 2.4467894221776447],
                [15, 4, -0.4782503630964142, 2.446667614844775],
                [16, 1, -2.3396869448373576, 3.3204881263460355],
                [16, 2, -3.2078226835720773, 3.3235070629897],
                [16, 3, -3.2079494303333775, 2.453629973388401],
                [16, 4, -2.3403853572098905, 2.4512062297969988]]
        return np.array(quad)

class QuadProb:
    '''
    Class :: Instantiate a QuadProb object that will allow us to calculate the
             probability content in a single quadrant.
    '''
    def __init__(self, RA, Dec):
        self.RA = RA
        self.Dec = Dec
        self.quadMap = {1: [0, 1, 2, 3], 2: [4, 5, 6, 7],
                        3: [8, 9, 10, 11], 4: [12, 13, 14, 15],
                        5: [16, 17, 18, 19], 6: [20, 21, 22, 23],
                        7: [24, 25, 26, 27], 8: [28, 29, 30, 31],
                        9: [32, 33, 34, 35], 10: [36, 37, 38, 39],
                        11: [40, 41, 42, 43], 12: [44, 45, 46, 47],
                        13: [48, 49, 50, 51], 14: [52, 53, 54, 55],
                        15: [56, 57, 58, 59], 16: [60, 61, 62, 63]}
        self.quadrant_size = np.array([3072, 3080])
        self.quadrant_scale = 1*u.arcsec

    def getWCS(self, quad_cents_RA, quad_cents_Dec):
        _wcs = WCS(naxis=2)
        _wcs.wcs.crpix = (self.quadrant_size + 1.0)/2
        _wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wrapped_quad_cent_Dec = max(min(quad_cents_Dec.value, 90), -90)
        _wcs.wcs.crval = [quad_cents_RA.value, wrapped_quad_cent_Dec]
        _wcs.wcs.cd = [[self.quadrant_scale.to(u.deg).value, 0],
                       [0, -self.quadrant_scale.to(u.deg).value]]
        return _wcs

def get_ztf_quadrants():
    """Calculate ZTF quadrant footprints as offsets from the telescope
    boresight."""
    quad_prob = QuadProb(0, 0)
    ztf_tile = ZTFtile(0, 0)
    quad_cents_ra, quad_cents_dec = ztf_tile.quadrant_centers()
    offsets = np.asarray([
        quad_prob.getWCS(
            quad_cents_ra[quadrant_id],
            quad_cents_dec[quadrant_id]
        ).calc_footprint(axes=quad_prob.quadrant_size)
        for quadrant_id in range(64)])
    return np.transpose(offsets, (2, 0, 1))

def get_quadrant_ipix(nside, ra, dec):

    quadrant_coords = get_ztf_quadrants()

    skyoffset_frames = SkyCoord(ra, dec, unit=u.deg).skyoffset_frame()
    quadrant_coords_icrs = SkyCoord(
                    *np.tile(
                        quadrant_coords[:, np.newaxis, ...],
                        (1, 1, 1)), unit=u.deg,
                    frame=skyoffset_frames[:, np.newaxis, np.newaxis]
                ).transform_to(ICRS)
    quadrant_xyz = np.moveaxis(
        quadrant_coords_icrs.cartesian.xyz.value, 0, -1)[0]

    ipixs = []
    for subfield_id, xyz in enumerate(quadrant_xyz):
        ipix = hp.query_polygon(nside, xyz)
        ipixs.append(ipix.tolist())
    return ipixs

