import plots
import numpy as np
import scipy.io as scio

print("Loading data")
pos_log = scio.loadmat("Data_sets/data_pos_no_jitter_Peter.mat")
pos_log = pos_log["pos_log"]

plots.positions(pos_log, 200)
