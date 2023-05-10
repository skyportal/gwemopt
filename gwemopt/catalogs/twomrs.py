import h5py
import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astroquery.vizier import Vizier
from scipy.special import gammaincinv

from gwemopt.catalogs.base_catalog import BaseCatalog


class TwoMRSCatalog(BaseCatalog):
    name = "2mrs"

    def download_catalog(self):
        (cat,) = Vizier.get_catalogs("J/ApJS/199/26/table3")

        ra, dec = cat["RAJ2000"], cat["DEJ2000"]
        magk = cat["Ktmag"]

        z = (u.Quantity(cat["cz"]) / c.c).to(u.dimensionless_unscaled)

        completeness = 0.5
        alpha = -1.0
        MK_star = -23.55
        MK_max = MK_star + 2.5 * np.log10(gammaincinv(alpha + 2, completeness))
        MK = magk - cosmo.distmod(z)
        idx = (z > 0) & (MK < MK_max)

        ra, dec = ra[idx], dec[idx]
        z = z[idx]
        magk = magk[idx]

        distmpc = cosmo.luminosity_distance(z).to("Mpc").value

        with h5py.File(self.get_catalog_path(), "w") as f:
            f.create_dataset("ra", data=ra)
            f.create_dataset("dec", data=dec)
            f.create_dataset("z", data=z)
            f.create_dataset("magk", data=magk)
            f.create_dataset("distmpc", data=distmpc)

    def load_catalog(self):
        with h5py.File(self.get_catalog_path(), "r") as f:
            ra, dec = f["ra"][:], f["dec"][:]
            z = f["z"][:]
            magk = f["magk"][:]
            distmpc = f["distmpc"][:]
        r = distmpc * 1.0
        mag = magk * 1.0

        return ra, dec, r, mag, z, distmpc
