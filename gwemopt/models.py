"""
Database schema.
"""

from flask_sqlalchemy import SQLAlchemy
import pkg_resources
from gwemopt.flaskapp import app

from astropy.io import ascii

db = SQLAlchemy(app)

def create_all(catalogFile):
    db.create_all()

    cat = ascii.read(catalogFile,format='ecsv')

    for row in cat.iterrows():
        print(row)
        print(stop)

        db.session.merge(Field(galaxy_id=int(galaxy_id),
                               ra=ra, dec=dec, dist=dist,
                               mag=mag))

    print(stop)



    refs = table.unique(table.Table.read(
        pkg_resources.resource_filename(__name__, 'input/ZTF.ref'),
        format='ascii', data_start=2, data_end=-1)['field', 'fid'])
    reference_images = {group[0]['field']: group['fid'].astype(int).tolist()
                        for group in refs.group_by('field').groups}

class Galaxy(db.Model):
    """Footprints and number of observations in each filter for standard PTF
    tiles"""

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

    mag = db.Column(
        db.Float,
        nullable=False,
        comment='B band mag of galaxy')


