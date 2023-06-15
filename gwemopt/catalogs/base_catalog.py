from pathlib import Path

import pandas as pd
from astroquery.vizier import Vizier

from gwemopt.paths import CATALOG_DIR

# Unset row limits when querying Vizier
Vizier.ROW_LIMIT = -1


class BaseCatalog:
    @property
    def name(self):
        """
        name for catalog files
        """
        raise NotImplementedError()

    def __init__(self, catalog_dir: Path = CATALOG_DIR):
        self.catalog_dir = catalog_dir

    def download_catalog(self):
        raise NotImplementedError

    def load_catalog(self):
        raise NotImplementedError

    def get_catalog_path(self, filetype="hdf5"):
        return CATALOG_DIR.joinpath(f"{self.name}.{filetype}")

    def get_catalog(self) -> pd.DataFrame:
        catalog_path = self.get_catalog_path()
        print(catalog_path)
        if not catalog_path.exists():
            self.download_catalog()
        else:
            print(f"Loading from saved catalog: {catalog_path}")
        return self.load_catalog()
