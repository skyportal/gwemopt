
import h5py
import astropy.io.fits

filename = '../catalog/CLU_20170106_galexwise_DaveUpdate.fits'
filename = '../catalog/CLU_20181213V2.fits'

hdul = astropy.io.fits.open(filename)
data = hdul[1].data

names = hdul[1].data.name
ras, decs = hdul[1].data.ra, hdul[1].data.dec
sfr, mstar = hdul[1].data.sfr_fuv, hdul[1].data.mstar
distmpc, magb = hdul[1].data.distmpc, hdul[1].data.magb
a, b2a, pa = hdul[1].data.a, hdul[1].data.b2a, hdul[1].data.pa
btc = hdul[1].data.btc

with h5py.File('../catalog/CLU.hdf5', 'w') as f:
    f.create_dataset('name', data=names)
    f.create_dataset('ra', data=ras)
    f.create_dataset('dec', data=decs)
    f.create_dataset('sfr_fuv', data=sfr)
    f.create_dataset('mstar', data=mstar)
    f.create_dataset('distmpc', data=distmpc)
    f.create_dataset('magb', data=magb)
    f.create_dataset('a', data=a)
    f.create_dataset('b2a', data=b2a)
    f.create_dataset('pa', data=pa)
    f.create_dataset('btc', data=btc)
