from numpy import log10, sqrt
import astroplan
from astropy.units import meter
from astropy.coordinates import EarthLocation
from gwemopt.paths import TESSELATION_DIR, REFS_DIR
from pathlib import Path
from numpy import loadtxt, empty, ndarray, empty, append as np_append, array
from gwemopt.utils import tesselation
from astropy import table
from typing import Any


class Telescope(astroplan.Observer):
    def __init__(
        self,
        telescope_name: str,
        telescope_description: dict,
        timezone="UTC",
        name=None,
        latitude=None,
        longitude=None,
        elevation=0 * meter,
        pressure=None,
        relative_humidity=None,
        temperature=None,
        description=None,
    ):
        self._name = telescope_name
        self._telescope_description = telescope_description
        self._tesselation = None
        self._referenceImage = None
        location = EarthLocation(
            self._telescope_description["longitude"],
            self._telescope_description["latitude"],
            self._telescope_description["elevation"],
        )
        super().__init__(
            location,
            timezone,
            name,
            latitude,
            longitude,
            elevation,
            pressure,
            relative_humidity,
            temperature,
            description,
        )

    @property
    def telescope_name(self) -> str:
        return self._name

    @property
    def telescope_mag(self) -> float:
        return float(self._telescope_description["magnitude"])

    @property
    def telescope_exptime(self) -> float:
        """
        Get the exposure time

        Returns
        -------
        float
            exposure time in seconds
        """
        return float(self._telescope_description["exposuretime"])

    @property
    def fov_type(self) -> str:
        return self._telescope_description["FOV_type"]

    @fov_type.setter
    def fov_type(self, new_fov_type):
        if new_fov_type not in ["square", "circle", "region"]:
            raise ValueError(
                f"Trying to assign a fov_type={new_fov_type}, Allowed fov_type are ['square', 'circle', 'region']. "
            )
        self.fov_type = new_fov_type

    @property
    def fov(self) -> float:
        """
        get the size of the field of view.

        Returns
        -------
        float
            field of view in degree
        """
        return float(self._telescope_description["FOV"])

    @fov.setter
    def fov(self, new_fov):
        self.fov = new_fov

    @property
    def fov_coverage(self) -> float:
        """
        get the size of the field of view.

        Returns
        -------
        float
            field of view in degree
        """
        return float(self._telescope_description["FOV_coverage"])

    @property
    def slew_rate(self) -> float:
        return float(self._telescope_description["slew_rate"])

    @property
    def readout(self) -> float:
        return float(self._telescope_description["readout"])

    @property
    def filters(self) -> str:
        return self._telescope_description["filt"]

    def mag_exposure(self, exposure_time: float) -> float:
        """
        Compute the magnitude limit reached for a given exposure time

        Parameters
        ----------
        exposure_time : float
            time of the exposure in seconds

        Returns
        -------
        float
            magnitude limit corrected for a given exposure time
        """
        return self.telescope_mag + (
            -2.5 * log10(sqrt(self.telescope_exptime / exposure_time))
        )

    def generate_tesselation(
        self,
        tessfile: Path | None = None,
        galaxy: bool = False,
        save_path: Path | None = None,
    ) -> ndarray:
        if not tessfile:
            try_tessfile = Path(
                TESSELATION_DIR.joinpath(self._telescope_description["tesselationFile"])
            )
            if try_tessfile.exists:
                tessfile = try_tessfile
        if tessfile and tessfile.exists():
            self._tesselation = loadtxt(tessfile, usecols=(0, 1, 2), comments="%")
        elif galaxy:
            self._tesselation = empty((3,))

        elif self.fov_type == "circle":
            self._tesselation = tesselation.tesselation_spiral(
                self.fov_type, self.fov, save_path=save_path
            )
        elif self.fov_type == "square":
            self._tesselation = tesselation.tesselation_packing(
                self.fov_type, self.fov, save_path=save_path
            )
        else:
            raise RuntimeError(
                f"""
tesselation error for telescope {self.telescope_name}, 
    either fov_type is different than 'square' or 'circle' (fov_type is {self.fov_type}),
    the galaxy argument is false or no tesselation file exist
"""
            )

    @property
    def tesselation(self) -> ndarray:
        if self._tesselation is None:
            self._tesselation == self.generate_tesselation()
        return self._tesselation

    @tesselation.setter
    def tesselation(self, tile_structs: dict[str, Any]):
        new_tess = empty((0, 3))
        tiles_struct = tile_structs[self.telescope_name]
        for index in tiles_struct.keys():
            ra, dec = tiles_struct[index]["ra"], tiles_struct[index]["dec"]
            new_tess = np_append(
                new_tess,
                [[index, ra, dec]],
                axis=0,
            )
        self._tesselation = array(new_tess)

    @property
    def referenceImages(self) -> dict:
        if not self._referenceImage and "referenceFile" in self._telescope_description:
            reffile = REFS_DIR.joinpath(self._telescope_description["referenceFile"])

            refs = table.unique(
                table.Table.read(reffile, format="ascii", data_start=2, data_end=-1)[
                    "field", "fid"
                ]
            )
            reference_images = {
                group[0]["field"]: group["fid"].astype(int).tolist()
                for group in refs.group_by("field").groups
            }
            reference_images_map = {0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "y"}
            for key in reference_images:
                reference_images[key] = [
                    reference_images_map.get(n, n) for n in reference_images[key]
                ]
            self._referenceImage = reference_images
        return self._referenceImage

    @property
    def sat_sun_restriction(self) -> float:
        return float(self._telescope_description.get("sat_sun_restriction", "0.0"))

    @property
    def overhead_per_exposure(self) -> float:
        return float(self._telescope_description.get("overhead_per_exposure", "0.0"))

    @property
    def filt_change_time(self) -> float:
        return float(self._telescope_description.get("filt_change_time", "0.0"))

    @property
    def min_observability_duration(self) -> float:
        return float(
            self._telescope_description.get("min_observability_duration", "0.0")
        )

    @property
    def horizon(self) -> float:
        return float(self._telescope_description.get("horizon"))

    @property
    def ha_constraint(self) -> tuple[float, float]:
        ha_constraint = self._telescope_description.get(
            "ha_constraint", "-24.0,24.0"
        ).split(",")
        return float(ha_constraint[0]), float(ha_constraint[1])

    @property
    def moon_constraint(self) -> float:
        return float(self._telescope_description.get("moon_constraint", "20.0"))

    @property
    def exposure_time(self) -> float:
        return float(self._telescope_description.get("exposuretime"))

    @property
    def dec_constraint(self) -> tuple[float, float] | None:
        dec_constraint = self._telescope_description.get("dec_constraint", None)
        if dec_constraint:
            dec_min, dec_max = dec_constraint.split(",")
            return float(dec_min), float(dec_max)
        else:
            return None

    def fov_center(self, galaxies_fov_sep: float) -> float:
        fov_center = self._telescope_description.get("FOV_center", self.fov)
        return fov_center * galaxies_fov_sep
