# gwemopt
Gravitational-wave Electromagnetic Optimization

Codebase plans:
1.) a code to interact with gracedb, download the skymaps, etc. I put a code in my github that does this, which could be improved.
2.) a code to create the tiling based on telescope configuration. I think everyone has a version of this, but in my mind, the code will have a number of options depending on how people prefer to do their tiling. So one python module with various versions, contributed by a few groups. I envision this should read in telescope configuration files, one for PS1, one for ZTF, etc. I don't know of anyone who has gone configuration file route, but this should be trivial. 
3.) a module to read in lightcurves from various sources, which I will contribute most likely, and enable their use in the optimization
4.) a module to generate exposure time as a function of tile, which accounts for number of hours available. This will be another module, and the different teams will once again contribute their own versions. This is the crux of the code where I hope the crossover is significant.
5.) optional: a module to actually do the scheduling based on when a tile is above the horizon, away from the moon, etc.
6.) a module to test the efficiency of the tiling, exposure time method choices, for a given lightcurve. I think most of us have versions of this, so one should probably be enough.

