# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
import os

import numpy as np
import matplotlib.pyplot as plt

# %% Global Parameters
RUN = False


# %% Functions


def steering_vectors2d(dir, theta, r, lambda_):
    e = dir * np.matrix([np.cos(theta), np.sin(theta)])
    return np.exp(-2j * (np.pi / lambda_) * e.T @ r)


def plot_directivity(W, N, title):
    # Calculating the directivity for a page in DFT-codebook
    beam = np.zeros((len(W), N))
    Theta = np.linspace(0, np.pi, N)
    # Sweep over range of angles, to calculate the angle with maximum "gain"
    for i in range(N):
        # Hardcode the array steering vector for a ULA with 10 elements
        A = np.exp(-1j * np.pi * np.cos(Theta[i]) * np.arange(0, len(W)))

        for j in range(len(W)):
            # The "gain" is found by multiplying the code-page with the steering vector
            beam[j, i] = np.abs(np.conjugate(W[j, :]).T @ A)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title(title)
    for j in range(len(W)):
        # Calculate the angle with max "gain".
        max_angle = np.pi - np.arccos(np.angle(W[j, 1]) / np.pi)

        # Plot the "gain"
        ax.plot(Theta, beam[j, :], label=f"{j}")
        ax.vlines(max_angle, 0, np.max(beam[j, :]),
                  colors='r', linestyles="dashed",
                  alpha=0.4)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=4)
    plt.show()


# %% main
if __name__ == "__main__":

    # Parameters:
    Nt = 10
    Nr = 10

    fc = 28e9  # Center frequncy
    lambda_ = 3e8 / fc

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

    print("Starts calculating")
    # Make ULA - Transmitter
    r_r = np.zeros((2, Nr))
    r_r[0, :] = np.linspace(0, (Nr - 1) * lambda_ / 2, Nr)

    # Make ULA - Receiver
    r_t = np.zeros((2, Nt))
    r_t[0, :] = np.linspace(0, (Nt - 1) * lambda_ / 2, Nt)

    # Empty arrays
    beam_t = np.zeros(np.shape(AoA)[0])
    beam_r = np.zeros(np.shape(AoA)[0])
    R = np.zeros((Nt, Nr, np.shape(AoA)[0]))

    # print("Calculating R")
    F = np.zeros((Nt, Nt), dtype=np.complex128)
    W = np.zeros((Nr, Nr), dtype=np.complex128)

    for p in range(Nt):
        F[p, :] = (1 / np.sqrt(Nt)) * np.exp(-1j * np.pi * np.arange(Nt) * ((2 * p - Nt) / Nt))

    for q in range(Nr):
        W[q, :] = (1 / np.sqrt(Nr)) * np.exp(-1j * np.pi * np.arange(Nr) * ((2 * q - Nr) / Nr))

    for j in range(np.shape(AoA)[0]):
        # print("Calculating H")
        alpha_rx = steering_vectors2d(dir=-1, theta=AoA[j, :], r=r_r, lambda_=lambda_)
        alpha_tx = steering_vectors2d(dir=1, theta=AoD[j, :], r=r_t, lambda_=lambda_)
        beta = coeff[j, :]
        H = np.zeros((Nr, Nt), dtype=np.complex128)

        for i in range(len(beta)):
            H += beta[i] * (alpha_rx[i].T @ np.conjugate(alpha_tx[i]))
        H = H*np.sqrt(Nr * Nt)

        for p in range(Nt):
            for q in range(Nr):
                R[p, q, j] = np.linalg.norm(np.sqrt(100000)*np.conjugate(W[q, :]).T @ H @ F[p, :])**2

        # print("Calculating Angles")
        index = np.unravel_index(np.argmax(R[:, :, j], axis=None), (Nt, Nr))

        # Equation 9 & 10 in "A Deep Learning Approach to Locatio"
        # Page 8
        beam_t[j] = np.arccos((2*index[0]-Nt)/Nt)*180/np.pi
        beam_r[j] = 180 - np.arccos((2*index[1]-Nr)/Nr)*180/np.pi
        # OLD: np.arccos(1-(2*index[1]/(Nr-1)))*180/np.pi

        # print(f"Reiver beam dir: {beam_r}\nTransmitter beam dir: {beam_t}")

    print("Done")

    # %% PLOT

    # Plot the directivity
    plot_directivity(W, 1000, "Receiver")
    plot_directivity(F, 1000, "Transmitter")

    # Plot the beam direction for the receiver and transmitter
    plt.figure()
    plt.title("Receiver")
    plt.plot(np.linspace(0, 2*np.pi, len(beam_r)),
             beam_r)

    plt.figure()
    plt.title("Transmitter")
    plt.plot(np.linspace(0, 2*np.pi, len(beam_t)),
             beam_t)
