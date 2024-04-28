import pandas as pd
from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
from astropy.table import Table
from astroquery.vizier import Vizier

from gwemopt.catalogs.base_catalog import BaseCatalog

# Unset row limits when querying Vizier
Vizier.ROW_LIMIT = -1


class GladeCatalog(BaseCatalog):
    name = "glade"

    def download_catalog(self):
        save_path = self.get_catalog_path()
        print(f"Downloading GLADE catalog to {save_path}...")
        (cat,) = Vizier.get_catalogs("VII/281/glade2", verbose=True)

        df = cat.to_pandas()

        key_map = {
            "RAJ2000": "ra",
            "DEJ2000": "dec",
            "Dist": "distmpc",
            "Bmag": "magb",
            "Kmag": "magk",
            "_2MASS": "2MASS",
        }

        copy_keys = ["GWGC", "PGC", "HyperLEDA", "z"]

        df = df[[x for x in key_map] + copy_keys]
        df.rename(columns=key_map, inplace=True)
        df["PGC"] = df["PGC"].astype(str)
        cat = Table.from_pandas(df)
        write_table_hdf5(cat, str(save_path), path="df")

    def load_catalog(self) -> pd.DataFrame:
        cat = read_table_hdf5(str(self.get_catalog_path()), path="df")
        df = cat.to_pandas()
        return df
