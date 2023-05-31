import numpy as np
from astropy.table import Table

from gwemopt.catalogs.base_catalog import BaseCatalog


class CluCatalog(BaseCatalog):
    name = "clu"

    def download_catalog(self):
        err = (
            f"CLU catalog is not available for automated download. "
            f"Please obtain a copy, and save it to {self.get_catalog_path()}."
        )
        raise FileNotFoundError(err)

    def load_catalog(self):
        cat = Table.read(self.get_catalog_path())

        ra, dec = cat["ra"], cat["dec"]
        sfr_fuv, mstar = cat["sfr_fuv"], cat["mstar"]
        distmpc, magb = cat["distmpc"], cat["magb"]
        a, b2a, pa = cat["a"], cat["b2a"], cat["pa"]
        btc = cat["btc"]

        idx = np.where(distmpc >= 0)[0]
        ra, dec = ra[idx], dec[idx]
        sfr_fuv, mstar = sfr_fuv[idx], mstar[idx]
        distmpc, magb = distmpc[idx], magb[idx]
        a, b2a, pa = a[idx], b2a[idx], pa[idx]
        btc = btc[idx]

        idx = np.where(~np.isnan(magb))[0]
        ra, dec = ra[idx], dec[idx]
        sfr_fuv, mstar = sfr_fuv[idx], mstar[idx]
        distmpc, magb = distmpc[idx], magb[idx]
        a, b2a, pa = a[idx], b2a[idx], pa[idx]
        btc = btc[idx]

        z = -1 * np.ones(distmpc.shape)
        r = distmpc * 1.0
        mag = magb * 1.0

        return ra, dec, r, mag, z, distmpc
