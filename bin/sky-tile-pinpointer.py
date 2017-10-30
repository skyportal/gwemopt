#!/usr/local/bin/python
# encoding: utf-8
"""
*Given a location on the sky, and a list of the centres of square sky-tiles, return the tile ID(s) that cover the sky position*

Useful for determining which telescope exposures cover a given point in the sky (assuming a square FOV)

:Author:
    David Young

:Date Created:
    June 26, 2017

Usage:
    sky-tile-pinpointer <tileSide> <ra> <dec> <pathToTileList>

ARGUMENTS
---------

    tileSide              the length of the side of the square sky-tile in degrees
    ra                    right ascension of location pinpoint within tile map (sexegesimal or decimal degrees)
    dec                   declination of location pinpoint within tile map (sexegesimal or decimal degrees)
    pathToTileList        path to the CSV list of tiles to match the location against (looks for 3 columns called `EXPID`, `RA`, `DEC`)


Options:
    -h, --help            show this help message
    -v, --version         show version
    -s, --settings        the settings file

Examples:

    ATLAS17aeu was a transient discovered by the ATLAS survey. It has coordiantes `09:13:13.89`, `+61:05:33.6`. ATLAS has a 5.46x5.46 deg FOV. Given a list of ATLAS exposures you can match the trasnient location against these exposures and output the exposure IDs that cover the location:

        python sky-tile-pinpointer.py 5.46 09:13:13.89 +61:05:33.6 G268556-ATLAS-coverage-stats.csv

    Output:

        TA136N62: 1.6166 deg from center (1.3090 N, -0.9490 E)
"""
################# GLOBAL IMPORTS ####################
import sys
import os
import math
import unicodecsv as csv
from fundamentals import tools
from astrocalc.coords import unit_conversion, separations
import healpy as hp
import numpy as np


def main(arguments=None):
    """
    *The main function used when ``sky-tile-pinpointer.py`` is run as a single script from the cl*
    """

    # MAKE SURE HEALPIX SMALL ENOUGH TO MATCH FOOTPRINTS CORRECTLY
    nside = 1024

    pi = (4 * math.atan(1.0))
    DEG_TO_RAD_FACTOR = pi / 180.0
    RAD_TO_DEG_FACTOR = 180.0 / pi

    # SETUP THE COMMAND-LINE UTIL SETTINGS
    su = tools(
        arguments=arguments,
        docString=__doc__,
        logLevel="WARNING",
        options_first=False,
        projectName=False
    )
    arguments, settings, log, dbConn = su.setup()

    # unpack remaining cl arguments using `exec` to setup the variable names
    # automatically
    for arg, val in arguments.iteritems():
        if arg[0] == "-":
            varname = arg.replace("-", "") + "Flag"
        else:
            varname = arg.replace("<", "").replace(">", "")
        if isinstance(val, str) or isinstance(val, unicode):
            exec(varname + " = '%s'" % (val,))
        else:
            exec(varname + " = %s" % (val,))
        if arg == "--dbConn":
            dbConn = val
        log.debug('%s = %s' % (varname, val,))

    tileSide = float(tileSide)

    # CONVERT RA AND DEC
    # ASTROCALC UNIT CONVERTER OBJECT
    converter = unit_conversion(
        log=log
    )
    ra = converter.ra_sexegesimal_to_decimal(
        ra=ra
    )
    dec = converter.dec_sexegesimal_to_decimal(
        dec=dec
    )

    # THE SKY-LOCATION AS A HEALPIXEL ID
    pinpoint = hp.ang2pix(nside, theta=ra, phi=dec, lonlat=True)

    matchedTileIds = []
    with open(pathToTileList, 'rb') as csvFile:
        csvReader = csv.DictReader(
            csvFile, dialect='excel', delimiter=',', quotechar='"')
        for row in csvReader:
            row["DEC"] = float(row["DEC"])
            row["RA"] = float(row["RA"])
            decCorners = (row["DEC"] - tileSide / 2,
                          row["DEC"] + tileSide / 2)
            corners = []
            for d in decCorners:
                if d > 90.:
                    d = 180. - d
                elif d < -90.:
                    d = -180 - d
                raCorners = (row["RA"] - (tileSide / 2) / np.cos(d * DEG_TO_RAD_FACTOR),
                             row["RA"] + (tileSide / 2) / np.cos(d * DEG_TO_RAD_FACTOR))
                for r in raCorners:
                    if r > 360.:
                        r = 720. - r
                    elif r < 0.:
                        r = 360. + r
                    corners.append(hp.ang2vec(r, d, lonlat=True))

            # FLIP CORNERS 3 & 4 SO HEALPY UNDERSTANDS POLYGON SHAPE
            corners = [corners[0], corners[1],
                       corners[3], corners[2]]

            # RETURN HEALPIXELS IN EXPOSURE AREA
            expPixels = hp.query_polygon(nside, np.array(
                corners))
            if pinpoint in expPixels:
                # CALCULATE SEPARATION IN ARCSEC
                calculator = separations(
                    log=log,
                    ra1=ra,
                    dec1=dec,
                    ra2=row["RA"],
                    dec2=row["DEC"],
                )
                angularSeparation, north, east = calculator.get()
                angularSeparation = float(angularSeparation) / 3600
                north = float(north) / 3600
                east = float(east) / 3600
                matchedTileIds.append(
                    row["EXPID"] + ": %(angularSeparation)1.4f deg from center (%(north)1.4f N, %(east)1.4f E)  " % locals())

    csvFile.close()

    for i in matchedTileIds:
        print i

    return

if __name__ == '__main__':
    main()

