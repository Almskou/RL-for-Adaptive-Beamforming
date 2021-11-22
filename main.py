# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

import os
import numpy as np
from oct2py import octave

octave.addpath(octave.genpath(f"{os.getcwd()}/Quadriga"))  # doctest: +SKIP
tmp = octave.our()
AoA = tmp[0]
AoD = tmp[1]
coeff = tmp[2]
