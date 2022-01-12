# Telescope Notes (March 23 2018)
For all the slew rate, we ignore most of the details (acceleration/deceleration, different directions, overhead/cooldown etc.) and use one number for it.
## ALTAS
15 deg/sec is used.
> The slew velocity is 15 deg/sec, and for moves smaller than 45 deg the time to slew and resume tracking is 6.5 ± 0.8 sec [source](https://arxiv.org/pdf/1802.00879.pdf).
## BlackGEM
0.5 deg/sec is used.

We have no information yet.
## LSST
6.3 deg/sec is used.

This comes from the LSST specification file (Telescope Mount Assembly Specifications Document LTS-103) and not yet tested. Minimum slew rate is used. The telescope moves differently in different directions. The speed is 7 deg/sec in azimuth direction and 3.5 deg/sec in elevation direction. Since the range in azimuth direction is 360 degrees and 90 in elevation direction, we use a 4:1 weighted average to represent a typical slew rate.
## PS1
0.6 deg/sec is used.
[source](https://rcuh.com/wp-content/uploads/2010/11/PSDC-350-004.pdf).
## ZTF
2.6 deg/sec is used.
> John Henning says 3 deg/sec top speed, and we’ve measured 2.9 deg/s in Dec, about 2.6 deg/s in HA so far (just saw some statistics on this.)  The acceleration time is not insignificant, however, and it takes about a 20 deg of slew to reach top speed, I believe.  An optimal controller could help minimize integrated slew time, but this might be a 20% kind of effect.
## IRIS
The limiting magnitude is estimated based on the aperture size. It could be larger. The exposure time is not found.
[Source](http://iris.lam.fr/le-telescope-2/les-caracteristiques-techniques-en-detail/)

## SVOM-MXT and SVOM-VT
Preliminary config files for both, as thoses config files are not usable yet with the code we keep the units (and infos) on each
parameters not to lose informations.
