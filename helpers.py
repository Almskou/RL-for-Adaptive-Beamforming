# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

import classes


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


def get_local_angle(AoA, Ori):
    # Calculate local AoA
    AoA_Local = (np.pi/2 - Ori)[:, np.newaxis] + AoA

    # Wrap where needed
    AoA_Local[AoA_Local < -np.pi] += 2*np.pi
    AoA_Local[AoA_Local > np.pi] -= 2*np.pi

    return AoA_Local


def get_data(RUN, ENGINE, pos_log_name, data_name, para):
    [fc, N, M, r_lim, stepsize, scenarios] = para

    # Load the data
    if not RUN:
        try:
            print("Loading data")
            pos_log = scio.loadmat(pos_log_name)
            pos_log = pos_log["pos_log"]

        except IOError:
            RUN = True
            print(f"Datafile {pos_log_name} not found")

        try:
            tmp = scio.loadmat(data_name)
            tmp = tmp["output"]

        except IOError:
            RUN = True
            print(f"Datafile {data_name} not found")

    if RUN:
        print("Creating track")
        # Create the class
        track = classes.Track(r_lim, stepsize)

        # Create the tracks
        pos_log = []
        for m in range(M):
            pos_log.append(track.run(N))

        # Save the data
        scio.savemat(pos_log_name, {"pos_log": pos_log, "scenarios": scenarios})

        if ENGINE == "octave":
            try:
                from oct2py import octave

            except ModuleNotFoundError:
                raise

            except OSError:
                raise OSError("'octave-cli' hasn't been added to path environment")

            print("Creating new data - octave")

            # Add Quadriga folder to octave path
            octave.addpath(octave.genpath(f"{os.getcwd()}/Quadriga"))

            # Run the scenario to get the simulated channel parameters
            if octave.get_data(fc, pos_log_name, data_name, ENGINE):
                try:
                    tmp = scio.loadmat(data_name)
                    tmp = tmp["output"]
                except FileNotFoundError:
                    raise FileNotFoundError(f"Data file {data_name} not loaded correctly")
            else:
                raise Exception("Something went wrong")

        elif ENGINE == "MATLAB":
            try:
                import matlab.engine
                print("Creating new data - MATLAB")

            except ModuleNotFoundError:
                raise Exception("You don't have matlab.engine installed")

            # start MATLAB engine
            eng = matlab.engine.start_matlab()

            # Add Quadriga folder to path
            eng.addpath(eng.genpath(f"{os.getcwd()}/Quadriga"))
            if eng.get_data(fc, pos_log_name, data_name, ENGINE):
                try:
                    tmp = scio.loadmat(data_name)
                    tmp = tmp["output"]

                except FileNotFoundError:
                    raise FileNotFoundError(f"Data file {data_name} not loaded correctly")

            else:
                raise Exception("Something went wrong")

            eng.quit()

        else:
            raise Exception("ENGINE name is incorrect")

    return tmp, pos_log
