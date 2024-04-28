import numpy as np
import pandas as pd
from astropy import constants as c
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
from astroquery.vizier import Vizier
from scipy.special import gammaincinv

from gwemopt.catalogs.base_catalog import BaseCatalog

# Unset row limits when querying Vizier
Vizier.ROW_LIMIT = -1


class TwoMRSCatalog(BaseCatalog):
    name = "2mrs"

    def download_catalog(self):
        save_path = self.get_catalog_path()
        print(f"Downloading 2MRS catalog to {save_path}...")

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
        cat = Table.from_pandas(df)
        write_table_hdf5(cat, str(save_path), path="df")

    def load_catalog(self) -> pd.DataFrame:
        cat = read_table_hdf5(str(self.get_catalog_path()), path="df")
        df = cat.to_pandas()
        return df
