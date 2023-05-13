#!/usr/bin/env python

# Copyright (C) 2023 Michael Coughlin, Robert Stein
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

""".
Gravitational-wave Electromagnetic Optimization
This script generates an optimized list of pointings and content for
reviewing gravitational-wave skymap likelihoods.
Comments should be e-mailed to michael.coughlin@ligo.org.
"""

import os
import sys
import warnings

import numpy as np

from gwemopt.run import run

if not os.getenv("DISPLAY", None):
    import matplotlib

    matplotlib.use("agg")


np.random.seed(0)

__author__ = "Michael Coughlin <michael.coughlin@ligo.org>"
__version__ = 1.0
__date__ = "6/17/2017"

# =============================================================================
#
#                               DEFINITIONS
#
# =============================================================================


warnings.filterwarnings("ignore")

run(sys.argv[1:])
