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
        df = Table.read(self.get_catalog_path()).to_pandas()

        mask = np.where(df["distmpc"] >= 0)[0]
        df = df.iloc[mask]

        mask = np.where(~np.isnan(df["magb"]))[0]
        df = df.iloc[mask]

        return df
