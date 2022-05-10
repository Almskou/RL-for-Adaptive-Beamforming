# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:34:07 2022

@author: almsk
"""

import numpy as np
N = 10_000
M = 8
chunksize = 10_000
Episodes = 1_000_000
idx_matrix = np.zeros([2, Episodes]).astype("int")

for e in range(Episodes):
    idx_matrix[:, e] = [np.random.randint(0, N - chunksize) if (N - chunksize) else 0, np.random.randint(0, M)]

np.save(f"Validation_idx/validation_idx_{chunksize}", idx_matrix)
