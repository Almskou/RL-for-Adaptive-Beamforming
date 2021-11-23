# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
import os
import numpy as np
import scipy

# %% Global Parameters
RUN = False


# %% Functions


def steering_vectors2d(theta, r, lambda_):
    e = [np.cos(theta), np.sin(theta)]
    return np.exp(-2j*(np.pi/lambda_)*e*r)


# %% main
if __name__ == "__main__":

    # Parameters:
    Nt = 1
    Nr = 1

    fc = 20e9  # Center frequncy
    lambda_ = 3e9/fc

    # Load the data
    if not RUN:
        try:
            print("Loading data")
            tmp = np.load("tmp.npy", allow_pickle=True)
        except IOError:
            RUN = True
            print("Datafile not found")

    if RUN:
        # Run the scenario to get the simulated channel parameters
        from oct2py import octave

        print("Creating new data")
        # Add Quadriga folder to octave path
        octave.addpath(octave.genpath(f"{os.getcwd()}/Quadriga"))
        tmp = octave.get_data(fc)
        np.save("tmp.npy", tmp)

    # Extract data
    print("Extracting data")
    AoA = tmp[0]  # Angle of Arrival
    AoD = tmp[1]  # Angle of Depature
    coeff = tmp[2]  # Channel Coeffiecents

    print("Calculating H")
    alpha_r = AoA[0, :]
    alpha_t = np.conjugate(AoA[0, :])
    beta = coeff[0, 0, :, 0]
    H = np.sum(np.sqrt(Nr*Nt)*beta*alpha_r*alpha_t)
    print("Done")
