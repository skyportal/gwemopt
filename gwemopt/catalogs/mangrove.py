import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

from gwemopt.catalogs.base_catalog import BaseCatalog

URL = "https://mangrove.lal.in2p3.fr/data/mangrove.hdf5"


class MangroveCatalog(BaseCatalog):
    name = "mangrove"

    def download_catalog(self):
        temp_path = self.get_temp_path()
        print(f"Mangrove catalog not found locally. Downloading to {temp_path}")
        subprocess.run(
            f"wget -O {self.get_temp_path()} {URL}",
            shell=True,
            check=True,
        )

        df = Table.read(temp_path).to_pandas()

        key_map = {
            "RA": "ra",
            "dist": "distmpc",
            "B_mag": "magb",
            "K_mag": "magk",
            "w1mpro": "magW1",
            "w2mpro": "magW2",
            "w3mpro": "magW3",
            "w4mpro": "magW4",
            "GWGC_name": "GWGC",
            "HyperLEDA_name": "HyperLEDA",
            "2MASS_name": "2MASS",
            "SDSS-DR12_name": "SDSS",
        }

        copy_keys = ["dec", "z", "stellarmass", "PGC", "AGN_flag"]

        df = df[[x for x in key_map] + copy_keys]
        df = df.rename(columns=key_map)

        for col in ["GWGC", "HyperLEDA", "2MASS", "SDSS"]:
            df[col] = df[col].astype(str)

        df.to_hdf(self.get_catalog_path(), key="df")
        temp_path.unlink()

    def get_temp_path(self):
        return self.get_catalog_path().with_stem(f"temp_{self.name}")

    def load_catalog(self) -> pd.DataFrame:
        df = pd.read_hdf(self.get_catalog_path(), key="df")

        mask = np.where(df["distmpc"] >= 0)[0]

        return df.iloc[mask]
