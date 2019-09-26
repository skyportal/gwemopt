import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS


def get_ccdcenters_radec(filename):
    hdul = fits.open(filename)
    nccds = len(hdul) - 1
    ccd_centers = dict()
    for i in range(1,nccds+1):
        w = WCS(hdul[i].header)
        ccdnum = hdul[i].header['CCDNUM']
        coords = np.array(w.wcs_pix2world(hdul[i].header['NAXIS1']/2,hdul[i].header['NAXIS2']/2,0))
        ccd_centers[str(ccdnum)] = coords
    hdul.close()
    return ccd_centers

def create_region_file(filename):
    hdul = fits.open(filename)
    nccds = len(hdul) - 1
    for i in range(1,nccds+1):
        w = WCS(hdul[i].header)
        w.footprint_to_file(filename='region_files/footprint_'+str(hdul[i].header['CCDNUM'])+'.reg')
    hdul.close()

def get_cd(filename):
    hdul = fits.open(filename) 
    w = WCS(hdul[1].header)
    hdul.close()
    return w.wcs.cd

def get_ccdcenters_xy(filename):
    "Calculates the x,y coordinates of the CCD center given the RA,Dec of the center and reference point.Uses Calabretta and Greisen(2002) eqns. 5,12,13,54"
    ccd_centers_xy = dict()
    phi_p = 180*u.deg
    ccd_centers_radec = get_ccdcenters_radec(filename)
    hdul = fits.open(filename)
    alpha_p = hdul[1].header['CRVAL1']*u.deg
    delta_p = hdul[1].header['CRVAL2']*u.deg
    ccdnums = ccd_centers_radec.keys()
    for ccdnum in ccdnums:
        alpha = ccd_centers_radec[ccdnum][0]*u.deg
        delta = ccd_centers_radec[ccdnum][1]*u.deg
        phi = phi_p + np.arctan2(-np.cos(delta)*np.sin(alpha-alpha_p),np.sin(delta)*np.cos(delta_p)-np.cos(delta)*np.sin(delta_p)*np.cos(alpha-alpha_p))
        theta = np.arcsin(np.sin(delta)*np.sin(delta_p) + np.cos(delta)*np.cos(delta_p)*np.cos(alpha-alpha_p))
        R_theta = (1*u.rad/np.tan(theta)).to(u.deg)
        x = R_theta * np.sin(phi)
        y = -R_theta * np.cos(phi)
        ccd_centers_xy[ccdnum] = [x.value,y.value]
    return ccd_centers_xy

def ccd_xy_to_radec(alpha_p,delta_p,ccd_centers_xy):
    # convert to Native longitude (phi) and latitude (theta)
    # need intermediate step (Rtheta) to deal with TAN projection
    # Calabretta & Greisen (2002), Eqn. 14,15
    alpha_p = alpha_p
    delta_p = delta_p
    ccd_centers_radec = dict()
    for ccdnum in ccd_centers_xy.keys():
        x = ccd_centers_xy[ccdnum][0]*u.deg
        y = ccd_centers_xy[ccdnum][1]*u.deg
        phi=np.arctan2(x,-y)
        Rtheta=np.sqrt(x**2+y**2)
	# now to theta using the TAN projection
	# Calabrett & Greisen (2002), Eqn. 55
        theta=np.arctan2(1,Rtheta.to(u.rad).value)*u.rad

	# Native longitude/latitude of fiducial point
	# appropriate for zenithal projections including TAN
        phi0=0*u.deg
        theta0=90*u.deg
	# Native longitude/latitue of celestial pole
	# for delta0<theta0 then phip should be 180 deg
	# and theta0=90 deg
        phi_p=180*u.deg
        theta_p=delta_p

	# Celestial longitude/latitude
	# Calabretta & Greisen (2002), Eqn. 2
        alpha=alpha_p+np.arctan2(-np.cos(theta)*np.sin(phi-phi_p),
                                 np.sin(theta)*np.cos(delta_p)-np.cos(theta)*np.sin(delta_p)*np.cos(phi-phi_p))
        delta=(np.arcsin(np.sin(theta)*np.sin(delta_p)+np.cos(theta)*np.cos(delta_p)*np.cos(phi-phi_p))).to(u.deg)
        alpha[alpha<0*u.deg]+=360*u.deg
        ccd_centers_radec[ccdnum] = np.array([alpha.value,delta.value])
    return ccd_centers_radec

def calculate_residuals(ccd_cent_code,ccd_cent_wcs):
    residuals = np.empty([len(ccd_cent_code),2])
    for i,ccdnum in enumerate(ccd_cent_code.keys()):
        residuals[i] = np.array([np.abs(ccd_cent_code[ccdnum][0]-ccd_cent_wcs[ccdnum][0]),np.abs(ccd_cent_code[ccdnum][1]-ccd_cent_wcs[ccdnum][1])])
    return residuals
        
