# gwemopt
Gravitational-wave Electromagnetic Optimization

[![Coverage Status](https://coveralls.io/repos/github/skyportal/gwemopt/badge.svg?branch=main)](https://coveralls.io/github/skyportal/gwemopt?branch=main)
[![CI](https://github.com/skyportal/gwemopt/actions/workflows/continous_integration.yml/badge.svg)](https://github.com/skyportal/gwemopt/actions/workflows/continous_integration.yml)
[![PyPI version](https://badge.fury.io/py/gwemopt.svg)](https://badge.fury.io/py/gwemopt)

### Citing gwemopt

When utilizing this code for a publication, kindly make a reference to the package by its name, gwemopt, and a citation to the software papers [Optimizing searches for electromagnetic counterparts of gravitational wave triggers](https://academic.oup.com/mnras/article/478/1/692/4987229) and [Teamwork Makes the Dream Work: Optimizing Multi-Telescope Observations of Gravitational-Wave Counterparts](https://academic.oup.com/mnras/article/489/4/5775/5565053). The BibTeX entry for the papers are:
```bibtex
@article{Coughlin:2018lta,
    author = "Coughlin, Michael W. and others",
    title = "{Optimizing searches for electromagnetic counterparts of gravitational wave triggers}",
    eprint = "1803.02255",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1093/mnras/sty1066",
    journal = "Mon. Not. Roy. Astron. Soc.",
    volume = "478",
    number = "1",
    pages = "692--702",
    year = "2018"
}
```
and
```bibtex
@article{Coughlin:2019qkn,
    author = "Coughlin, Michael W. and others",
    title = "{Optimizing multitelescope observations of gravitational-wave counterparts}",
    eprint = "1909.01244",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1093/mnras/stz2485",
    journal = "Mon. Not. Roy. Astron. Soc.",
    volume = "489",
    number = "4",
    pages = "5775--5783",
    year = "2019"
}
```

and for the ability to balance field exposures.

```bibtex
@article{Almualla:2020hbs,
    author = "Almualla, Mouza and Coughlin, Michael W. and Anand, Shreya and Alqassimi, Khalid and Guessoum, Nidhal and Singer, Leo P.",
    title = "{Dynamic Scheduling: Target of Opportunity Observations of Gravitational Wave Events}",
    eprint = "2003.09718",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.1093/mnras/staa1498",
    month = "3",
    year = "2020"
}
```

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
gwemopt-run ....
```

where ... corresponds to the various arguments. 
