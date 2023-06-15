import numpy as np
import pandas as pd
from astropy import constants as c
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astroquery.vizier import Vizier
from scipy.special import gammaincinv

from gwemopt.catalogs.base_catalog import BaseCatalog


class TwoMRSCatalog(BaseCatalog):
    name = "2mrs"

    def download_catalog(self):
        cat = Vizier.get_catalogs("J/ApJS/199/26/table3")[0]

        cat["z"] = (u.Quantity(cat["cz"]) / c.c).to(u.dimensionless_unscaled)
        df = cat.to_pandas()
        df = df[df["z"] > 0]
        key_map = {
            "RAJ2000": "ra",
            "DEJ2000": "dec",
            "Ktmag": "magk",
        }

        copy_keys = ["z"]

        df = df[[x for x in key_map] + copy_keys]
        df.rename(columns=key_map, inplace=True)
        df["distmpc"] = cosmo.luminosity_distance(df["z"].to_numpy()).to("Mpc").value

        completeness = 0.5
        alpha = -1.0
        mk_star = -23.55
        mk_max = mk_star + 2.5 * np.log10(gammaincinv(alpha + 2, completeness))
        mk = df["magk"].to_numpy() - cosmo.distmod(df["z"].to_numpy()).value

        mask = mk < mk_max
        df = df[mask]
        df.to_hdf(self.get_catalog_path(), key="df")

    def load_catalog(self):
        df = pd.read_hdf(self.get_catalog_path(), key="df")
        return df
