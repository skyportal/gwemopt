
import h5py
import astropy.io.fits

filename = '../catalog/CLU_20170106_galexwise_DaveUpdate.fits'

hdul = astropy.io.fits.open(filename)
data = hdul[1].data

names = hdul[1].data.NAME
ras, decs = hdul[1].data.RA, hdul[1].data.DEC
sfr, mstar = hdul[1].data.SFR_FUV, hdul[1].data.MSTAR
distmpc, magb = hdul[1].data.DISTMPC, hdul[1].data.MAGB

with h5py.File('../catalog/CLU.hdf5', 'w') as f:
    f.create_dataset('NAME', data=names)
    f.create_dataset('RA', data=ras)
    f.create_dataset('DEC', data=decs)
    f.create_dataset('SFR_FUV', data=sfr)
    f.create_dataset('MSTAR', data=mstar)
    f.create_dataset('DISTMPC', data=distmpc)
    f.create_dataset('MAGB', data=magb)

