# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
import os
import sys

import numpy as np
import scipy.io as scio

import classes
import plots

from numba import njit


# %% Functions
def steering_vectors2d(direction, theta, r, lambda_):
    """
    Calculates the steering vector for a Standard ULA, with given antenna positions
    wave length and angles.
    Can be used for both receivers and transmitters by changing the direction parameter.
    :param direction: -1 for impinging direction, 1 for propagation direction
    :param theta: Angles in radians
    :param r: Coordinate positions of antennas in meters.
    :param lambda_: Wave length of carrier wave in meters
    :return: Array steering vector
    """
    e = direction * np.matrix([np.cos(theta), np.sin(theta)])
    return np.exp(-2j * (np.pi / lambda_) * e.T @ r)


def codebook(Nb, N):
    """
    Calculates the codebook based on the number of antennae and beams
    :param Nb: Number of beams
    :param N: Number of antennae
    :return: Codebook matrix
    """
    Cb = np.zeros((Nb, N), dtype=np.complex128)
    for n in range(Nb):
        Cb[n, :] = ((1 / np.sqrt(N)) * np.exp(-1j * np.pi * np.arange(N) * ((2 * n - Nb) / (Nb))))

    return Cb


def angle_to_beam(AoA, W):
    beam_tmp = np.zeros([len(W), 1])
    beam = np.zeros([len(AoA), 1])

    for i in range(len(AoA)):
        A = (1 / np.sqrt(len(W[0, :]))) * np.exp(-1j * np.pi * np.cos(AoA[i]) * np.arange(0, len(W[0, :])))
        for j in range(len(W)):
            # The gain is found by multiplying the code-page with the steering vector
            beam_tmp[j] = np.abs(np.conjugate(W[j, :]).T @ A)

        beam[i] = np.argmax(beam_tmp)
    return beam


def get_local_angle(AoA, Ori):
    """
    Transforms angles in global Quadriga coordinate system,
    to angles in local antenna array coordinate system.
    :param AoA: Angles in global coordinate system
    :param Ori: Orientation of the antenna array in global coordinate system
    :return: Angles in local coordinate system
    """
    # Calculate local AoA
    AoA_Local = (np.pi / 2 - Ori)[:, np.newaxis] + AoA

    # Wrap where needed
    AoA_Local[AoA_Local < -np.pi] += 2 * np.pi
    AoA_Local[AoA_Local > np.pi] -= 2 * np.pi

    return AoA_Local


def discrete_ori(Ori, N):
    angles = [((n + 1) * np.pi) / N for n in range(N - 1)]

    Ori_abs = np.abs(Ori)
    Ori_discrete = np.zeros(np.shape(Ori))

    for n in range(1, N - 1):
        Ori_discrete[np.logical_and(Ori_abs > angles[n - 1],
                                    Ori_abs <= angles[n])] = n

    Ori_discrete[Ori_abs > angles[-1:]] = N - 1

    return Ori_discrete


def discrete_angle(pos, N):
    Angle = np.arctan2(pos[1, :], pos[0, :])
    # Angle: [0 deg, 360 deg] in radians
    Angle[Angle < 0] += 2 * np.pi

    # Discrete angles
    angles = [(((n + 1) * 2 * np.pi) / N) for n in range(N - 1)]

    Angle_discrete = np.zeros(np.shape(Angle))

    for n in range(1, N - 1):
        Angle_discrete[np.logical_and(Angle > angles[n - 1],
                                      Angle <= angles[n])] = n

    Angle_discrete[Angle > angles[-1:]] = N - 1

    return Angle_discrete


def discrete_dist(pos, N, r_lim):
    pos_norm = np.linalg.norm(pos[0:2, :], axis=0)
    base = int(r_lim / N)
    return (base * np.round(pos_norm / base)).astype(int)


def misalignment_prob(R_db, R_max_db, x_db):
    # Create zeros vector with shape of R
    tmp = np.zeros(np.shape(R_db))

    # All places where the R values is less
    # than R_max - x_db is set to 1.
    tmp[R_db < R_max_db - x_db] = 1

    # Return the x_db misalignment probability
    return np.mean(tmp)


def noisy_ori(ori_vector):
    # "smooting" factor in random walk filter
    K = 21
    new_orientation = np.empty_like(ori_vector)
    for idx, episode in enumerate(ori_vector):
        z_axis = episode[0][2]

        # generate the random walk
        a_bar = np.zeros(len(z_axis) + K)
        for i in range(len(z_axis) + K - 1):
            a_bar[i + 1] = a_bar[i] + np.random.normal(0, 0.5 * np.pi / 180)

        # generate the additive orientation noise as MA filtered random walk
        a = np.zeros(len(z_axis))
        for i in range(len(z_axis)):
            a[i] = np.sum(a_bar[i:i + K]) / K

        # Add the noise to original data and wrap angles to range [-pi:pi] for correct signs
        res_z_axis = z_axis + a
        for i in range(len(a)):
            while res_z_axis[i] > np.pi:
                res_z_axis[i] -= 2 * np.pi

            while res_z_axis[i] < -1 * np.pi:
                res_z_axis[i] += 2 * np.pi

        res = np.zeros((3, len(z_axis)))
        res[2] = res_z_axis
        new_orientation[idx, 0] = res

    return new_orientation


def get_data(RUN, ENGINE, case, multi_user, pos_log_name, data_name, para):
    """
    Generates parameters for the channel model.
    Parameters are either loaded from earlier simulations,
    or generated from Quadriga simulations.
    Either a MATLAB or Octave engine is used to run simulations.
    :param RUN: Bool to determine if load from files or run simulation
    :param ENGINE: Which engine to use for simulations. "MATLAB" or "Octave"
    :param pos_log_name: Name of data file containing positions and scenarios eg: "data_pos.mat"
    :param data_name: Name of data file containing parameters/coefficients from simulations eg: "data.mat"
    :param para: List of simulation settings/parameters used in the simulations
    :return:
    """
    [fc, N, M, r_lim, intersite, sample_period, scenarios, debug] = para

    # Load the data
    if not RUN:
        try:
            print("Loading data")
            pos_log = scio.loadmat("Data_sets/" + pos_log_name)
            pos_log = pos_log["pos_log"].T

        except IOError:
            RUN = True
            print(f"Datafile {pos_log_name} not found")

        try:
            tmp = scio.loadmat("Data_sets/" + data_name)
            tmp = tmp["output"]

        except IOError:
            RUN = True
            print(f"Datafile {data_name} not found")

    if RUN:
        print("Creating track")

        # Create the class
        track = classes.Track(case=case, delta_t=sample_period, intersite=intersite,
                              r_lim=r_lim, debug_print=debug[1])
        pos_bs = track.pos_bs

        pos_log_done = False
        while pos_log_done is False:
            # Create the tracks
            pos_log = [pos_bs]
            for m in range(M):
                pos_log.append(track.run(N))

            if debug[0]:
                plots.positions(pos_log[1:], pos_bs, r_lim, show=True, save=False)
                user_input = input("Does the created track(s) look fine (yes/no/stop)")
                if user_input.lower() == "yes":
                    pos_log_done = True
                if user_input.lower() == "stop":
                    sys.exit("Program stopped by user")
            else:
                pos_log_done = True

        print('track done')
        # Save the data
        scio.savemat("Data_sets/" + pos_log_name, {"pos_log": pos_log, "scenarios": scenarios})

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
            if multi_user:
                if octave.get_data_multi_user(fc, pos_log_name, data_name, ENGINE):
                    try:
                        tmp = scio.loadmat("Data_sets/" + data_name)
                        tmp = tmp["output"]
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Data file {data_name} not loaded correctly")
                else:
                    raise Exception("Something went wrong")
            else:
                if octave.get_data_multi_env(fc, pos_log_name, data_name, ENGINE):
                    try:
                        tmp = scio.loadmat("Data_sets/" + data_name)
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
            if multi_user:
                if eng.get_data_multi_user(fc, pos_log_name, data_name, ENGINE):
                    try:
                        tmp = scio.loadmat("Data_sets/" + data_name)
                        tmp = tmp["output"]
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Data file {data_name} not loaded correctly")
                else:
                    raise Exception("Something went wrong")
            else:
                if eng.get_data_multi_env(fc, pos_log_name, data_name, ENGINE):
                    try:
                        tmp = scio.loadmat("Data_sets/" + data_name)
                        tmp = tmp["output"]
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Data file {data_name} not loaded correctly")
                else:
                    raise Exception("Something went wrong")

            eng.quit()

        else:
            raise Exception("ENGINE name is incorrect")

    return tmp, pos_log


@njit()
def jit_H(beta, alpha_rx, alpha_tx, Nr, Nt):
    H = np.zeros((Nr, Nt), dtype=np.complex128)
    for i in range(len(beta)):
        H += beta[i] * np.dot(alpha_rx[i, :].T, np.conjugate(alpha_tx[i, :]))
    H = H * np.sqrt(Nr * Nt)

    return H


@njit()
def jit_Reward(H, F, W, P_t):
    R = np.zeros((len(F[:, 0]), len(W[:, 0])), dtype=np.complex128)
    # Calculate the reward
    for p in range(len(F[:, 0])):  # p - transmitter
        for q in range(len(W[:, 0])):  # q - receiver
            R[p, q] = np.dot(np.dot(np.conjugate(W[q, :]).T, H), F[p, :]) * np.sqrt(P_t)

    return R
