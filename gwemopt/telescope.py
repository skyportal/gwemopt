from numpy import log10, sqrt
import astroplan
from astropy.units import meter
from astropy.coordinates import EarthLocation
from gwemopt.paths import TESSELATION_DIR, REFS_DIR
from pathlib import Path
from numpy import loadtxt, empty, ndarray
from gwemopt.utils import tesselation
from astropy import table


class Telescope(astroplan.Observer):
    def __init__(
        self,
        telescope_name: str,
        _telescope_description: dict,
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
        self._telescope_description = _telescope_description
        self._tesselation = None
        self._referenceField = None
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
    def referenceFields(self) -> dict:
        if not self._referenceField and "referenceFile" in self._telescope_description:
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
            self._referenceField = reference_images
        return self._referenceField

    @property
    def tesselation(self) -> ndarray:
        if not self._tesselation:
            self._tesselation == self.generate_tesselation()
        return self._tesselation
