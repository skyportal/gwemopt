import healpy as hp
import ligo.skymap.plot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import get_body
from astropy.time import Time

cmap = "cylon"

plt.ioff()

matplotlib.rcParams.update({"font.size": 16})
matplotlib.rcParams["contour.negative_linestyle"] = "solid"

UNIT = "Gravitational-wave probability"
CBAR_BOOL = False


def add_edges():
    """
    Add longitude and latitude lines to a healpix map.
    """
    hp.graticule()
    plt.grid(True)
    lons = np.arange(-150.0, 180, 30.0)
    lats = np.zeros(lons.shape)
    for lon, lat in zip(lons, lats):
        hp.projtext(lon, lat, "%.0f" % lon, lonlat=True)
    lats = np.arange(-60.0, 90, 30.0)
    lons = np.zeros(lons.shape)
    for lon, lat in zip(lons, lats):
        hp.projtext(lon, lat, "%.0f" % lat, lonlat=True)


def add_sun_moon(params, ax):
    """
    Add sun and moon position to the skymap
    """
    # plot sun and moon position at the beginning of the observation
    start_time = params["gpstime"]
    start_time = Time(start_time, format="gps", scale="utc")

    sun_position = get_body("sun", start_time)
    moon_position = get_body("moon", start_time)

    ax.plot(
        sun_position.ra,
        sun_position.dec,
        transform=ax.get_transform("world"),
        color="yellow",
    )
    ax.plot(
        moon_position.ra,
        moon_position.dec,
        transform=ax.get_transform("world"),
        color="grey",
    )

    # also plot (in smaller scale) sun and moon position for the 7 following days
    # This allows us to show the path of both sun and moon for the coming days

    dt = 21600  # 1/4 day

    for i in range(1, 29):  # 29 is 4*7
        new_time = params["gpstime"] + (dt * i)

        new_time = Time(new_time, format="gps", scale="utc")

        new_moon_position = get_body("moon", new_time)
        ax.plot(
            new_moon_position.ra,
            new_moon_position.dec,
            transform=ax.get_transform("world"),
            color="black",
            marker=".",
            markersize=1,
        )

        if not i % 8:
            # only plot point for the sun every two days in order to avoid overlap
            new_sun_position = get_body("sun", new_time)
            ax.plot(
                new_sun_position.ra,
                new_sun_position.dec,
                transform=ax.get_transform("world"),
                color="black",
                marker=".",
                markersize=1,
            )
