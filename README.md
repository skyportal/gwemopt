# gwemopt
Gravitational-wave Electromagnetic Optimization

[![Coverage Status](https://coveralls.io/repos/github/skyportal/gwemopt/badge.svg?branch=main)](https://coveralls.io/github/skyportal/gwemopt?branch=main)
[![CI](https://github.com/skyportal/gwemopt/actions/workflows/continous_integration.yml/badge.svg)](https://github.com/skyportal/gwemopt/actions/workflows/continous_integration.yml)
[![PyPI version](https://badge.fury.io/py/gwemopt.svg)](https://badge.fury.io/py/gwemopt)

The code currently can:
- interact with gracedb, download the skymaps, read them etc. 
- read telescope configuration files with location, FOV, limiting magnitude, exposure times, etc.
- create the tiling based on telescope configuration (MOC, Shaon's method).
- generate exposure time as a function of tile, which accounts for number of hours available. 
- perform scheduling to include most of the requested exposures
- test the efficiency of the tiling, exposure time method choices, for a given lightcurve.

Current planned improvements / open questions:
- Include moon and sky brightness when scheduling
- How to improve scheduling when multiple (and different numbers of exposures are expected)
- How to use WAW when inclination is not immediately available
- How to include distance estimates as function of sky location in PEM

Related repositories:
- https://github.com/JavedRANA/emgwfollowup

- https://github.com/shaonghosh/sky_tiling.git

- https://github.com/omsharansalafia/waw

- https://github.com/manleongchan/optimizationcode 

References:
- Rana et al: http://iopscience.iop.org/article/10.3847/1538-4357/838/2/108/meta

- Salafia et al: https://arxiv.org/abs/1704.05851

- Ghosh et al: https://www.aanda.org/articles/aa/abs/2016/08/aa27712-15/aa27712-15.html

- Chan et al: http://iopscience.iop.org/article/10.3847/1538-4357/834/1/84/meta

- Coughlin and Stubbs: https://link.springer.com/article/10.1007/s10686-016-9503-4 

# Setting up the environment

If you want the latest version, we recommend creating a clean environment:

```commandline
conda create -n gwemopt python=3.11
git clone git@github.com:skyportal/gwemopt.git
pip install -e gwemopt
pre-commit install
```

or if you just want the latest version on Github:

```commandline
pip install gwemopt
```

If you run into dependency issues, you can try installing dependencies via conda:

```
conda install numpy scipy matplotlib astropy h5py shapely
conda install -c astropy astroquery
conda install -c conda-forge voeventlib astropy-healpix python-ligo-lw ligo-segments ligo.skymap ffmpeg
```

And then run `pip install -e gwemopt` again.

# Usage

Once installed, You can use gwemopt via the command line:

```commandline
python -m gwemopt ....
```

where ... corresponds to the various arguments. 
