# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
import os

import numpy as np

# %% Global Parameters
RUN = False


# %% Functions


def steering_vectors2d(dir, theta, r, lambda_):
    e = dir * np.matrix([np.cos(theta), np.sin(theta)])
    return np.exp(-2j * (np.pi / lambda_) * e.T @ r)


# %% main
if __name__ == "__main__":

    # Parameters:
    Nt = 10
    Nr = 10

    fc = 20e9  # Center frequncy
    lambda_ = 3e9 / fc

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

    # Make ULA
    N = 10  # Number of antennas
    r = np.zeros((2, N))
    r[0, :] = np.linspace(0, (N - 1) * lambda_ / 2, N)

    print("Calculating H")
    alpha_rx = steering_vectors2d(dir=-1, theta=AoA[0, :], r=r, lambda_=lambda_)
    alpha_tx = steering_vectors2d(dir=1, theta=AoD[0, :], r=r, lambda_=lambda_)
    beta = coeff[0, :]
    H = np.zeros((Nr, Nr), dtype=np.complex128)
    for i in range(len(beta)):
        H += np.sqrt(Nr * Nt) * beta[i] * (alpha_rx[i].T @ np.conjugate(alpha_tx[i]))
    print("Done")

    F = np.zeros((Nt, Nt), dtype=np.complex128)
    W = np.zeros((Nr, Nr), dtype=np.complex128)

    for p in range(Nt):
        F[p, :] = (1 / np.sqrt(Nt)) * np.exp(-1j * np.pi * np.arange(Nt) * (2 * p - 1 - Nt) / Nt)

    for q in range(Nr):
        W[q, :] = (1 / np.sqrt(Nr)) * np.exp(-1j * np.pi * np.arange(Nr) * (2 * q - 1 - Nr) / Nr)

    R = np.zeros((Nt, Nr))
    for p in range(Nt):
        for q in range(Nr):
            R[p, q] = np.linalg.norm(np.sqrt(100000)*np.conjugate(W[q, :]).T @ H @ F[p, :])**2
