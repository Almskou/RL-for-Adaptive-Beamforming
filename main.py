# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from time import time

import helpers

# %% Global Parameters
RUN = True
ENGINE = "octave"  # "octave" OR "MATLAB"

# %% main
if __name__ == "__main__":

    # Number of steps in a episode
    N = 50

    # Radius for commuication range [m]
    r_lim = 200

    # Stepsize limits [min, max]
    stepsize = [0.5, 5]

    # Number of episodes
    M = 2

    # Number of antennae
    Nt = 10  # Transmitter
    Nr = 10  # Receiver

    fc = 28e9  # Center frequncy
    lambda_ = 3e8 / fc

    # Possible scenarios
    scenarios = ['3GPP_38.901_UMi_LOS']  # '3GPP_38.901_UMi_NLOS'

    t_start = time()
    # Load or create the data
    tmp, pos_log = helpers.get_data(RUN, ENGINE,
                                    "data_pos.mat", "data.mat",
                                    [fc, N, M, r_lim, stepsize, scenarios])
    print(f"Took: {time()-t_start}")
    M = len(pos_log)

    # Extract data
    print("Extracting data")
    AoA_Global = tmp[0][0]  # Angle of Arrival
    AoD_Global = tmp[1][0]  # Angle of Depature
    coeff = tmp[2][0]  # Channel Coeffiecents
    Orientation = tmp[3][0]  # Orientation

    print("Starts calculating")
    # Make ULA - Transmitter
    r_r = np.zeros((2, Nr))
    r_r[0, :] = np.linspace(0, (Nr - 1) * lambda_ / 2, Nr)

    # Make ULA - Receiver
    r_t = np.zeros((2, Nt))
    r_t[0, :] = np.linspace(0, (Nt - 1) * lambda_ / 2, Nt)

    # Empty arrays
    beam_t = np.zeros((np.shape(AoA_Global)[0], N))
    beam_r = np.zeros((np.shape(AoA_Global)[0], N))
    R = np.zeros((np.shape(AoA_Global)[0], Nt, Nr, N))
    AoA_Local = []

    # print("Calculating R")
    F = np.zeros((Nt, Nt), dtype=np.complex128)
    W = np.zeros((Nr, Nr), dtype=np.complex128)

    for p in range(Nt):
        F[p, :] = ((1 / np.sqrt(Nt)) * np.exp(-1j * np.pi *
                   np.arange(Nt) * ((2 * p - Nt) / Nt)))

    for q in range(Nr):
        W[q, :] = (1 / np.sqrt(Nr)) * np.exp(-1j * np.pi * np.arange(Nr) * ((2 * q - Nr) / Nr))

    for episode in range(np.shape(AoA_Global)[0]):
        AoA_Local.append(helpers.get_local_angle(AoA_Global[episode][0], Orientation[episode][0][2, :]))
        for j in range(np.shape(AoA_Global[episode][0])[0]):
            # print("Calculating H")
            alpha_rx = helpers.steering_vectors2d(dir=-1, theta=AoA_Local[episode][j, :], r=r_r, lambda_=lambda_)
            alpha_tx = helpers.steering_vectors2d(dir=1, theta=AoD_Global[episode][0][j, :], r=r_t, lambda_=lambda_)
            beta = coeff[episode][0][j, :]

            H = np.zeros((Nr, Nt), dtype=np.complex128)
            for i in range(len(beta)):
                H += beta[i] * (alpha_rx[i].T @ np.conjugate(alpha_tx[i]))
                H = H*np.sqrt(Nr * Nt)

            for p in range(Nt):
                for q in range(Nr):
                    R[episode, p, q, j] = np.linalg.norm(np.sqrt(100000)*np.conjugate(W[q, :]).T @ H @ F[p, :])**2

            # print("Calculating Angles")
            index = np.unravel_index(np.argmax(R[episode, :, :, j], axis=None), (Nt, Nr))

            # Equation 9 & 10 in "A Deep Learning Approach to Location"
            # Page 8
            beam_t[episode, j] = np.arccos((2*index[0]-Nt)/Nt)*180/np.pi
            beam_r[episode, j] = 180 - np.arccos((2*index[1]-Nr)/Nr)*180/np.pi
            # OLD: np.arccos(1-(2*index[1]/(Nr-1)))*180/np.pi

            # print(f"Reiver beam dir: {beam_r}\nTransmitter beam dir: {beam_t}")

    print("Done")

    # %% PLOT

    # Plot the directivity
    helpers.plot_directivity(W, 1000, "Receiver")
    helpers.plot_directivity(F, 1000, "Transmitter")

    # Plot the beam direction for the receiver and transmitter
    # Caculate the Line Of Sight angle for transmitter in GLOBAL coordinate system
    AoA_LOS_t = np.abs(np.arctan2(pos_log[0][1], pos_log[0][0]) * 180/np.pi)

    # Caculate the Line Of Sight angle for receiver in GLOBAL coordinate system
    AoA_LOS_r_GLOBAL = np.arctan2(-pos_log[0][1], -pos_log[0][0])
    AoA_LOS_r_GLOBAL.shape = (N, 1)

    # Caculate the Line Of Sight angle for receiver in LOCAL coordinate system
    AoA_LOS_r_LOCAL = helpers.get_local_angle(AoA_LOS_r_GLOBAL[0][0], Orientation[0][0][2, :]) * 180/np.pi

    plt.figure()
    plt.title("Receiver")
    plt.plot(beam_r[0, :], label="Beam")
    plt.plot(np.abs(AoA_LOS_r_LOCAL), label="LOS")
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
        ax.plot(pos_log[m][0, :], pos_log[m][1, :])

    ax.set_xlim([-r_lim, r_lim])
    ax.set_ylim([-r_lim, r_lim])
    plt.show()
