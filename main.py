# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
from time import time

import matplotlib.pyplot as plt

import numpy as np

import helpers
import plots

# %% Global Parameters
RUN = False
ENGINE = "MATLAB"  # "octave" OR "MATLAB"
FILENAME = "38.901_UMi_LOS_20000_5_0.5_1"  # After the "data_" or "data_pos_"

# %% main
if __name__ == "__main__":

    # Number of steps in a episode
    N = 20000

    # Radius for communication range [m]
    r_lim = 200

    # Stepsize limits [m] [min, max]
    stepsize = [0.5, 5]

    # Number of episodes
    M = 2

    # Number of antennae
    Nt = 4  # Transmitter
    Nr = 4  # Receiver

    # Number of beams
    Nbt = 5  # Transmitter
    Nbr = 5  # Receiver

    fc = 28e9  # Center frequency
    lambda_ = 3e8 / fc  # Wave length
    P_t = 10000  # Transmission power

    # Possible scenarios for Quadriga simulations
    scenarios = ['3GPP_38.901_UMi_LOS']  # '3GPP_38.901_UMi_NLOS'

    t_start = time()
    # Load or create the data
    tmp, pos_log = helpers.get_data(RUN, ENGINE,
                                    f"data_pos_{FILENAME}.mat", f"data_{FILENAME}",
                                    [fc, N, M, r_lim, stepsize, scenarios])
    print(f"Took: {time() - t_start}")

    # Re-affirm that "M" matches data
    M = len(pos_log)

    # Extract data from Quadriga simulation
    print("Extracting data")
    AoA_Global = tmp[0][0]  # Angle of Arrival in Global coord. system
    AoD_Global = tmp[1][0]  # Angle of Departure in Global coord. system
    coeff = tmp[2][0]  # Channel Coefficients
    Orientation = tmp[3][0]  # Orientation in Global coord. system

    print("Starts calculating")
    # Make ULA antenna positions - Transmitter
    r_r = np.zeros((2, Nr))
    r_r[0, :] = np.linspace(0, (Nr - 1) * lambda_ / 2, Nr)

    # Make ULA antenna positions - Receiver
    r_t = np.zeros((2, Nt))
    r_t[0, :] = np.linspace(0, (Nt - 1) * lambda_ / 2, Nt)

    # Preallocate empty arrays
    beam_t = np.zeros((M, N))
    beam_r = np.zeros((M, N))
    R = np.zeros((M, Nt, Nr, N))
    AoA_Local = []
    F = np.zeros((Nbt, Nt), dtype=np.complex128)
    W = np.zeros((Nbr, Nr), dtype=np.complex128)

    # Calculate DFT-codebook - Transmitter
    for p in range(Nbt):
        F[p, :] = ((1 / np.sqrt(Nt)) * np.exp(-1j * np.pi * np.arange(Nt) * ((2 * p - Nbt) / (Nbt))))

    # Calculate DFT-codebook - Receiver
    for q in range(Nbr):
        W[q, :] = (1 / np.sqrt(Nr)) * np.exp(-1j * np.pi * np.arange(Nr) * ((2 * q - Nbr) / (Nbr)))

    for m in range(M):
        print(f"Progress: {(m / M) * 100}%")
        AoA_Local.append(helpers.get_local_angle(AoA_Global[m][0], Orientation[m][0][2, :]))
        for j in range(np.shape(AoA_Global[m][0])[0]):  # Episodes might have different lengths

            # Calculate steering vectors for transmitter and receiver
            alpha_rx = helpers.steering_vectors2d(direction=-1, theta=AoA_Local[m][j, :],
                                                  r=r_r, lambda_=lambda_)
            alpha_tx = helpers.steering_vectors2d(direction=1, theta=AoD_Global[m][0][j, :],
                                                  r=r_t, lambda_=lambda_)

            # Calculate channel matrix H
            beta = coeff[m][0][j, :]
            H = np.zeros((Nr, Nt), dtype=np.complex128)
            for i in range(len(beta)):
                H += beta[i] * (alpha_rx[i].T @ np.conjugate(alpha_tx[i]))
            H = H * np.sqrt(Nr * Nt)

            # Calculate received power for all beam-pairs
            for p in range(Nt):
                for q in range(Nr):
                    R[m, p, q, j] = np.linalg.norm(np.sqrt(P_t) * np.conjugate(W[q, :]).T @ H @ F[p, :]) ** 2

            # Determine the index of the beam-pair with highest gain
            index = np.unravel_index(np.argmax(R[m, :, :, j], axis=None), (Nt, Nr))

            # Equation 9 & 10 in "A Deep Learning Approach to Location"
            # Page 8
            beam_t[m, j] = np.arccos((2 * index[0] - Nt) / Nt) * 180 / np.pi
            beam_r[m, j] = 180 - np.arccos((2 * index[1] - Nr) / Nr) * 180 / np.pi

    print("Progress: 100%")
    print("Done")

    # %% PLOT

    # Plot the directivity
    plots.directivity(W, 1000, "Receiver")
    plots.directivity(F, 1000, "Transmitter")

    # Plot the beam direction for the receiver and transmitter
    # Caculate the Line Of Sight angle for transmitter in GLOBAL coordinate system
    AoA_LOS_t = np.abs(np.arctan2(pos_log[0][1], pos_log[0][0]) * 180 / np.pi)

    # Caculate the Line Of Sight angle for receiver in GLOBAL coordinate system
    AoA_LOS_r_GLOBAL = np.arctan2(-pos_log[0][1], -pos_log[0][0])
    AoA_LOS_r_GLOBAL.shape = (N, 1)

    # Caculate the Line Of Sight angle for receiver in LOCAL coordinate system
    AoA_LOS_r_LOCAL = np.abs(helpers.get_local_angle(AoA_LOS_r_GLOBAL[0][0], Orientation[0][0][2, :]) * 180 / np.pi)

    plt.figure()
    plt.title("Receiver")
    plt.plot(beam_r[0, :], label="Beam")
    plt.plot(AoA_LOS_r_LOCAL, label="LOS")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Transmitter")
    plt.plot(beam_t[0, :], label="Beam")
    plt.plot(AoA_LOS_t, label="LOS")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title("Kunst")
    ax.add_patch(plt.Circle((0, 0), r_lim, color='r', alpha=0.1))
    for m in range(M):
        ax.plot(pos_log[m][0, :], pos_log[m][1, :], label=f"{m}")

    ax.set_xlim([-r_lim, r_lim])
    ax.set_ylim([-r_lim, r_lim])
    plt.legend()
    plt.show()
