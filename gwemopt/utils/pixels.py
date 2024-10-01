import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, SkyCoord
from mocpy import MOC
from tqdm import tqdm


def get_region_moc(ra, dec, regions, max_depth=12, n_threads=None):
    for reg in regions:
        ra_tmp = reg.vertices.ra
        dec_tmp = reg.vertices.dec

        coords = np.stack([np.array(ra_tmp), np.array(dec_tmp)])

    skyoffset_frames = SkyCoord(ra, dec, unit=u.deg).skyoffset_frame()
    coords_icrs = SkyCoord(
        *np.tile(coords[:, np.newaxis, ...], (1, 1, 1)),
        unit=u.deg,
        frame=skyoffset_frames[:, np.newaxis, np.newaxis],
    ).transform_to(ICRS)

    mocs = []
    for ccd_coords in tqdm(coords_icrs):
        stacked = np.stack((ccd_coords.ra.deg, ccd_coords.dec.deg), axis=1)
        result = stacked.reshape(-1, ccd_coords.ra.deg.shape[1])
        lon_lat_list = [row for row in result]
        mocs.append(sum(MOC.from_polygons(lon_lat_list, False, 10, n_threads)))

    return mocs
