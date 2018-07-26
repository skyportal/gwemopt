"""
Database schema.
"""

from flask_sqlalchemy import SQLAlchemy
import pkg_resources
from gwemopt.flaskapp import app
import numpy as np
import healpy as hp
from astropy.table import Table
from astropy.io import ascii

db = SQLAlchemy(app)

def create_all(catalogFile):
    db.drop_all()
    db.create_all()

    nside = 256

    cat = ascii.read(catalogFile,format='ecsv')
    ipixs = hp.ang2pix(nside, cat['RAJ2000'].to('deg').value, cat['DEJ2000'].to('deg').value, lonlat=True)

    for ii,row in enumerate(cat):
        if np.mod(ii,10000) == 0:
            print('%d/%d'%(ii,len(cat)))
        ra, dec, dist = float(row["RAJ2000"]), float(row["DEJ2000"]), float(row["Dist"])
        Bmag, Kmag = float(row["Bmag"]), float(row["Kmag2"])
        ipix = ipixs[ii]

        db.session.merge(Galaxy(galaxy_id=int(ii),
                               ra=ra, dec=dec, dist=dist,
                               Bmag=Bmag, Kmag=Kmag,
                               ipix=int(ipix)))
    db.session.commit()

def read_catalog():

    galaxies = Galaxy.query.all()
    cat = Table(names=('RAJ2000', 'DEJ2000', 'Dist', 'Bmag', 'Kmag', 'ipix'), dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'i8'))
    for galaxy in galaxies:
        gal = [galaxy.ra, galaxy.dec, galaxy.dist, galaxy.Bmag, galaxy.Kmag, galaxy.ipix]
        cat.add_row(gal)

class Galaxy(db.Model):
    """Galaxy information"""

    galaxy_id = db.Column(
        db.Integer,
        primary_key=True,
        comment='Galaxy field ID')

    ra = db.Column(
        db.Float,
        nullable=False,
        comment='RA of galaxy')

    dec = db.Column(
        db.Float,
        nullable=False,
        comment='Dec of galaxy')

    dist = db.Column(
        db.Float,
        nullable=False,
        comment='Distance of galaxy')

    Bmag = db.Column(
        db.Float,
        nullable=False,
        comment='B band mag of galaxy')

    Kmag = db.Column(
        db.Float,
        nullable=False,
        comment='K band mag of galaxy')

    ipix = db.Column(
        db.Integer,
        comment='Healpix index')
