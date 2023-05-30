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

"""
This module contains the DECamtile class, which is used to calculate decam chip gaps
"""

import astropy
import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.wcs import WCS


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
        self.ccd_size = np.array([2046, 4094])
        #        self.quadrant_scale=0.27*u.arcsec

        if isinstance(RA, astropy.units.quantity.Quantity):
            self.RA = RA
        else:
            self.RA = RA * u.deg
        if isinstance(Dec, astropy.units.quantity.Quantity):
            self.Dec = Dec
        else:
            self.Dec = Dec * u.deg

        self.number = number

        centers = self.ccd_centers()
        self.ccd_RA = np.array([centers[key][0] for key in centers.keys()]) * u.deg
        self.ccd_Dec = np.array([centers[key][1] for key in centers.keys()]) * u.deg
        cd = np.array(
            [[-1.60128078e-07, 7.28644399e-05], [-7.28804334e-05, -1.41273307e-07]]
        )

        self.missing_ccds = []
        if missing is not None and len(missing) > 0:
            for m in missing:
                if isinstance(m, int) or len(m) == 1:
                    # assume this is 0 origin
                    self.missing_ccds.append(m)
        #                elif len(m)==2:
        # this assumes CCD and quadrant numbers are 1-origin
        #                    self.missing_quadrants.append(4*(m[0]-1)+(m[1]-1))
        good_ccds = np.setdiff1d(
            np.arange(len(self.ccd_RA)), np.array(self.missing_ccds)
        )

        self._wcs = []
        for i in good_ccds:
            self._wcs.append(self.ccd_WCS(cd, self.ccd_RA[i], self.ccd_Dec[i]))

    def ccd_WCS(self, cd, ccd_RA, ccd_Dec):
        """
        w=ccd_WCS(cd,crpix,ccd_RA, ccd_Dec)
        returns a WCS object that is specific to the ccd RA and Dec specified
        overall scale, size are determined by class variables
        """

        w = WCS(naxis=2)
        w.wcs.crpix = (self.ccd_size + 1.0) / 2
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [ccd_RA.value, max(min(ccd_Dec.value, 90), -90)]
        w.wcs.cd = cd
        return w

    def ccd_centers(self):
        """
        alpha,delta=ccd_centers()
        return celestial coordinates for the centers of each ccd
        given the pointing center RA,Dec
        """
        ccd_centers_radec = ccd_xy_to_radec(self.RA, self.Dec, self.get_ccd_centers())
        return ccd_centers_radec

    def get_ccd_centers(self):
        ccd_cent_xy = {
            "1": [-0.30848869614381674, 0.9048868132934124],
            "3": [0.3126638732505829, 0.903736681246994],
            "4": [-0.46424313388432176, 0.7410089796458399],
            "5": [-0.15364984859501787, 0.7408713142103732],
            "6": [0.1571387259082878, 0.7402269739588593],
            "7": [0.467603613257108, 0.7392171229774838],
            "8": [-0.6198946868359682, 0.5768134685376444],
            "9": [-0.30955498335035025, 0.5768085523046568],
            "10": [0.0013477793249647779, 0.5763113502103236],
            "11": [0.3121902019689574, 0.5755750278008702],
            "12": [0.6226930307061053, 0.5743855934179499],
            "13": [-0.7755501585170581, 0.4124544450213818],
            "14": [-0.46558119678543103, 0.41251128825254235],
            "15": [-0.15464112242830574, 0.41229197585181315],
            "16": [0.15656398568088345, 0.4114606530655467],
            "17": [0.4674414894955094, 0.41072825795192136],
            "18": [0.7777331864109138, 0.409567932442902],
            "19": [-0.7759834336369162, 0.24831566877921676],
            "20": [-0.46607502185988525, 0.24782905120552876],
            "21": [-0.15509932243638994, 0.24758285798181506],
            "22": [0.15626265250880025, 0.24702609588409785],
            "23": [0.4674434728526691, 0.2461625655342552],
            "24": [0.777482221168825, 0.24521534351199428],
            "25": [-0.9314559066649502, 0.08383480908831277],
            "26": [-0.6216994874177798, 0.08353996510188882],
            "27": [-0.31102353489225215, 0.08283960916939065],
            "28": [0.0005071693155911257, 0.08241792995217975],
            "29": [0.31158305482235177, 0.08163431022316624],
            "30": [0.6224533719596026, 0.08098146395460736],
            "31": [0.9320777373043198, 0.08044138862297144],
            "32": [-0.9319464165707169, -0.0805589216843629],
            "33": [-0.622133652371652, -0.08122227572955273],
            "34": [-0.31148773321326756, -0.08193206187722929],
            "35": [-0.00014499352178874166, -0.08254844513458418],
            "36": [0.31098135984096015, -0.08317938096258874],
            "37": [0.6217138680647583, -0.08356797238031384],
            "38": [0.9315841692181391, -0.08418488728808847],
            "39": [-0.7773609424381199, -0.245412819661229],
            "40": [-0.46728294158166866, -0.24630563002879535],
            "41": [-0.15615176138477885, -0.2469488238588537],
            "42": [0.15523767984355089, -0.24768250766829886],
            "43": [0.4662187887667616, -0.24809293832066254],
            "44": [0.776246372428461, -0.24844509764265615],
            "45": [-0.7775470321940903, -0.40967055522534995],
            "46": [-0.4676908153361856, -0.41077195744348843],
            "47": [-0.15653339105153957, -0.41170581376861437],
            "48": [0.15449971354932118, -0.41233044750284964],
            "49": [0.46565358463600437, -0.41260284266708563],
            "50": [0.7757206230994781, -0.41269011164669245],
            "51": [-0.6225729237947966, -0.5746346822329046],
            "52": [-0.3122155321260558, -0.5755970743456301],
            "53": [-0.0011401216990000642, -0.5763197415054745],
            "54": [0.30968238386568164, -0.5769630649576271],
            "55": [0.6200804504434518, -0.5769886473852198],
            "56": [-0.46754566383013146, -0.7392890744530034],
            "57": [-0.15717538834988146, -0.7402009474290661],
            "58": [0.15377123111958424, -0.7408432843903323],
            "59": [0.4642380873704591, -0.7410163636305219],
            "60": [-0.3125067998457982, -0.9038265779307155],
            "62": [0.3086296599756487, -0.9050452408374547],
        }
        return ccd_cent_xy


def get_decam_ccds(ra, dec, save_footprint=False):
    """Calculate DECam CCD footprints as offsets from the telescope
    boresight. Also optionally saves the footprints in a region file
    called footprint.reg"""
    #    ccd_prob = CCDProb(ra, dec)
    decam_tile = DECamtile(ra, dec)
    ccd_cents_ra = decam_tile.ccd_RA
    ccd_cents_dec = decam_tile.ccd_Dec
    offsets = np.asarray(
        [
            decam_tile._wcs[ccd_id].calc_footprint(axes=decam_tile.ccd_size)
            for ccd_id in range(len(ccd_cents_ra))
        ]
    )
    if save_footprint:
        ra, dec = np.transpose(offsets, (2, 0, 1))
        with open("footprint.reg", "a") as f:
            for i in range(len(ra)):
                lines = [
                    "# Region file format: DS9 version 4.0 \n",
                    '# global color=green font="helvetica 12 bold select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source \n',
                    "ICRS \n",
                    "polygon("
                    + str(ra[i, 0])
                    + ","
                    + str(dec[i, 0])
                    + ","
                    + str(ra[i, 1])
                    + ","
                    + str(dec[i, 1])
                    + ","
                    + str(ra[i, 2])
                    + ","
                    + str(dec[i, 2])
                    + ","
                    + str(ra[i, 3])
                    + ","
                    + str(dec[i, 3])
                    + ") # color=green, width=2 \n",
                ]
                f.writelines(lines)
            f.close()
    return np.transpose(offsets, (2, 0, 1))


def get_decam_quadrant_ipix(nside, ra, dec):
    ccd_coords = get_decam_ccds(0, 0)

    skyoffset_frames = SkyCoord(ra, dec, unit=u.deg).skyoffset_frame()
    ccd_coords_icrs = SkyCoord(
        *np.tile(ccd_coords[:, np.newaxis, ...], (1, 1, 1)),
        unit=u.deg,
        frame=skyoffset_frames[:, np.newaxis, np.newaxis],
    ).transform_to(ICRS)
    ccd_xyz = np.moveaxis(ccd_coords_icrs.cartesian.xyz.value, 0, -1)[0]

    ipixs = []
    for xyz in ccd_xyz:
        ipix = hp.query_polygon(nside, xyz)
        ipixs.append(ipix.tolist())
    return ipixs


def ccd_xy_to_radec(alpha_p, delta_p, ccd_centers_xy):
    # convert to Native longitude (phi) and latitude (theta)
    # need intermediate step (Rtheta) to deal with TAN projection
    # Calabretta & Greisen (2002), Eqn. 14,15
    alpha_p = alpha_p
    delta_p = delta_p
    ccd_centers_radec = dict()
    for ccdnum in ccd_centers_xy.keys():
        x = ccd_centers_xy[ccdnum][0] * u.deg
        y = ccd_centers_xy[ccdnum][1] * u.deg
        phi = np.arctan2(x, -y)
        Rtheta = np.sqrt(x**2 + y**2)
        # now to theta using the TAN projection
        # Calabrett & Greisen (2002), Eqn. 55
        theta = np.arctan2(1, Rtheta.to(u.rad).value) * u.rad

        # Native longitude/latitue of celestial pole
        # for delta0<theta0 then phip should be 180 deg
        # and theta0=90 deg
        phi_p = 180 * u.deg

        # Celestial longitude/latitude
        # Calabretta & Greisen (2002), Eqn. 2
        alpha = alpha_p + np.arctan2(
            -np.cos(theta) * np.sin(phi - phi_p),
            np.sin(theta) * np.cos(delta_p)
            - np.cos(theta) * np.sin(delta_p) * np.cos(phi - phi_p),
        )
        delta = (
            np.arcsin(
                np.sin(theta) * np.sin(delta_p)
                + np.cos(theta) * np.cos(delta_p) * np.cos(phi - phi_p)
            )
        ).to(u.deg)
        alpha[alpha < 0 * u.deg] += 360 * u.deg
        ccd_centers_radec[ccdnum] = np.array([alpha.value, delta.value])
    return ccd_centers_radec
