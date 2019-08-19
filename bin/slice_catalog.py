
import h5py
import astropy.io.fits
from astropy.table import Table
from astropy import units as u

filename = '../catalog/CLU_20170106_galexwise_DaveUpdate.fits'
filename = '../catalog/CLU_20181213V2.fits'
filename = '../catalog/CLU_20190406_filled.hdf5'
filename = '../catalog/CLU_20190708_marshalFormat.hdf5'

t = Table.read(filename)
columns = ['name', 'ra', 'dec', 'distmpc', 'sfr_fuv', 'mstar',
           'magb', 'a', 'b2a', 'pa', 'btc']
t.keep_columns(columns)

t['ra'].unit = u.deg
t['dec'].unit = u.deg
t['distmpc'].unit = u.Mpc

t.write('../catalog/CLU.hdf5', path='/data', format='hdf5', serialize_meta=True, overwrite=True)

t2 = t[:10]

t2.write('../catalog/CLU_mini.hdf5', path='/data', format='hdf5', serialize_meta=True, overwrite=True)

print(t2)
print(stop)

hdul[1].data = hdul[1].data[:10]

names = hdul[1].data.name
ras, decs = hdul[1].data.ra, hdul[1].data.dec
sfr, mstar = hdul[1].data.sfr_fuv, hdul[1].data.mstar
distmpc, magb = hdul[1].data.distmpc, hdul[1].data.magb
a, b2a, pa = hdul[1].data.a, hdul[1].data.b2a, hdul[1].data.pa
btc = hdul[1].data.btc

t = Table([names, ras, decs, sfr, mstar, distmpc, magb,
           a, b2a, pa, btc],
          names=('names', 'ras', 'decs', 'sfr', 'mstar', 'distmpc', 'magb',
                 'a', 'b2a', 'pa', 'btc'))
t['ras'].unit = u.deg
t['decs'].unit = u.deg
t['distmpc'].unit = u.Mpc

#t.write('../catalog/CLU_astropy.hdf5', path='/data', format='hdf5', serialize_meta=True)

#t = Table.read('../catalog/CLU_astropy.hdf5')

with h5py.File('../catalog/CLU_mini.hdf5', 'w') as f:
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
