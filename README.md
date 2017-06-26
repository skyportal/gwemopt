# gwemopt
Gravitational-wave Electromagnetic Optimization

The code currently can:
- interact with gracedb, download the skymaps, read them etc. 
- read telescope configuration files with location, FOV, limiting magnitude, exposure times, etc.

The code needs to:
- create the tiling based on telescope configuration (MOC, Shaon's method).
- generate exposure time as a function of tile, which accounts for number of hours available. 
- do the scheduling based on when a tile is above the horizon, away from the moon, etc.
- test the efficiency of the tiling, exposure time method choices, for a given lightcurve.

Related repositories:
- https://github.com/shaonghosh/sky_tiling.git

- https://github.com/omsharansalafia/waw

- https://github.com/manleongchan/optimizationcode 

References:
- Rana et al: http://iopscience.iop.org/article/10.3847/1538-4357/838/2/108/meta

- Salafia et al: https://arxiv.org/abs/1704.05851

- Ghosh et al: https://www.aanda.org/articles/aa/abs/2016/08/aa27712-15/aa27712-15.html

- Chan et al: http://iopscience.iop.org/article/10.3847/1538-4357/834/1/84/meta

- Coughlin and Stubbs: https://link.springer.com/article/10.1007/s10686-016-9503-4 

