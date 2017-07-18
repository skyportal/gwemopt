# gwemopt
Gravitational-wave Electromagnetic Optimization

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

