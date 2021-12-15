# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
from time import time

import numpy as np
import json

import helpers
import classes
import plots

# %% Global Parameters
RUN = False
ENGINE = "MATLAB"  # "octave" OR "MATLAB"
METHOD = "SARSA"  # "simple", "SARSA" OR "Q-LEARNING"
ADJ = True
ORI = True  # Include the orientiation in the state
DIST = False  # Include the dist in the state
LOCATION = False   # Include location in polar coordinates in the state
FILENAME = "test_line"  # After the "data_" or "data_pos_"
CASE = "walk"  # "walk" or "car"

# %% main
if __name__ == "__main__":

    # Load Scenario configuration
    with open(f'Cases/{CASE}.json', 'r') as fp:
        case = json.load(fp)

    # State parameters
    n_actions = 2
    n_ori = 2

    dist_res = 8
    angle_res = 8

    # Number of steps in a episode
    N = 30000

    # Chunk size (More "episodes" per episode)
    chunksize = 30000

    # Number of episodes
    M = 1

    # Number of episodes per chunk
    Episodes = 10

    # Radius for communication range [m]
    r_lim = case["rlim"]

    # Stepsize limits [m] [min, max]
    stepsize = [case["stepsize"]["min"],
                case["stepsize"]["max"]]

    # Probability for during a big turn
    change_dir = case["change_dir"]

    # Number of antennae
    Nt = case["transmitter"]["antennea"]  # Transmitter
    Nr = case["receiver"]["antennea"]  # Receiver

    # Number of beams
    Nbt = case["transmitter"]["beams"]  # Transmitter
    Nbr = case["receiver"]["beams"]  # Receiver

    fc = case["fc"]  # Center frequency
    lambda_ = 3e8 / fc  # Wave length
    P_t = case["P_t"]  # Transmission power

    # Possible scenarios for Quadriga simulations
    scenarios = ['3GPP_38.901_UMi_LOS']  # '3GPP_38.901_UMi_NLOS'

    t_start = time()
    # Load or create the data
    tmp, pos_log = helpers.get_data(RUN, ENGINE,
                                    f"data_pos_{FILENAME}.mat", f"data_{FILENAME}",
                                    [fc, N, M, r_lim, stepsize, scenarios, change_dir])
    print(f"Took: {time() - t_start}")

    if len(pos_log[0, 0, :]) > N:
        pos_log = pos_log[:, :, 0:N]

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
    AoA_Local = []

    # Calculate DFT-codebook - Transmitter
    F = helpers.codebook(Nbt, Nt)

    # Calculate DFT-codebook - Receiver
    W = helpers.codebook(Nbr, Nr)

    # Calculate the AoA in the local coordinate system
    for m in range(M):
        AoA_Local.append(helpers.get_local_angle(AoA_Global[m][0], Orientation[m][0][2, :]))

    # Create the Environment
    Env = classes.Environment(W, F, Nt, Nr,
                              r_r, r_t, fc, P_t)

    # Create action space
    action_space = np.arange(Nbr)

    # Create the discrete orientation if ORI is true
    if ORI:
        ori_discrete = np.zeros([M, N])
        for m in range(M):
            ori_discrete[m, :] = helpers.discrete_ori(Orientation[m][0][2, :], Nbr)
    else:
        ori_discrete = None

    if DIST or LOCATION:
        dist_discrete = np.zeros([M, N])
        for m in range(M):
            dist_discrete[m, :] = helpers.discrete_dist(pos_log[m], dist_res, r_lim)
    else:
        dist_discrete = None

    if LOCATION:
        angle_discrete = np.zeros([M, N])
        for m in range(M):
            angle_discrete[m, :] = helpers.discrete_angle(pos_log[m], angle_res)
    else:
        angle_discrete = None

    # Number of chuncks
    nchunk = int(np.floor(N/chunksize))

    # RUN
    action_log = np.zeros([Episodes, chunksize])
    R_log = np.zeros([Episodes, chunksize])
    R_max_log = np.zeros([Episodes, chunksize])
    R_min_log = np.zeros([Episodes, chunksize])
    R_mean_log = np.zeros([Episodes, chunksize])

    for episode in range(Episodes):
        print(f"Progress: {(episode / Episodes) * 100:0.2f}%")
        # Create the Agent
        Agent = classes.Agent(action_space, eps=0.1, alpha=["constant", 0.7])

        # Create the State
        State_tmp = []
        State_tmp.append(list(np.random.randint(0, Nbr, n_actions)))

        if DIST or LOCATION:
            State_tmp.append(list([dist_discrete[0]]))
        else:
            State_tmp.append(["N/A"])

        if ORI:
            State_tmp.append(list(np.random.randint(0, Nbr, n_ori)))
        else:
            State_tmp.append(["N/A"])

        if LOCATION:
            State_tmp.append(list([angle_discrete[0]]))
        else:
            State_tmp.append(["N/A"])

        State = classes.State(State_tmp)

        # Choose data
        data_idx = np.random.randint(0, N-chunksize)
        path_idx = np.random.randint(0, M)

        # Update the enviroment data
        Env.update_data(AoA_Local[path_idx][data_idx:data_idx+chunksize],
                        AoD_Global[path_idx][0][data_idx:data_idx+chunksize],
                        coeff[path_idx][0][data_idx:data_idx+chunksize])

        # Initiate the action
        action = np.random.choice(action_space)

        end = False
        # Run the episode
        for n in range(chunksize):
            if ORI:
                ori = int(ori_discrete[path_idx, data_idx + n])
                if n < chunksize-1:
                    next_ori = int(ori_discrete[path_idx, data_idx + n + 1])
            else:
                ori = None
                next_ori = None

            if DIST or LOCATION:
                dist = dist_discrete[path_idx, data_idx + n]
                if n < chunksize-1:
                    next_dist = dist_discrete[path_idx, data_idx + n + 1]
            else:
                dist = None
                next_dist = None

            if LOCATION:
                angle = angle_discrete[path_idx, data_idx + n]
                if n < chunksize-1:
                    next_angle = angle_discrete[path_idx, data_idx + n + 1]
            else:
                angle = None
                next_angle = None

            if n == chunksize-1:
                end = True

            para = [dist, ori, angle]
            para_action = [next_dist, ori, angle]
            para_next = [next_dist, next_ori, next_angle]

            State.update_state(action, para=para)

            if ADJ:
                action = Agent.e_greedy_adj(State.get_state(para=para), action)
            else:
                action = Agent.e_greedy(State.get_state(para=para))

            R, R_max, R_min, R_mean = Env.take_action(n, action)

            if METHOD == "simple":
                Agent.update(State, action, R, para=para)
            elif METHOD == "SARSA":
                if ADJ:
                    next_action = Agent.e_greedy_adj(State.get_nextstate(action,
                                                                         para_next=para_action), action)
                else:
                    next_action = Agent.e_greedy(State.get_nextstate(action,
                                                                     para_next=para_action))
                Agent.update_sarsa(R, State, action,
                                   next_action,
                                   para_next=para_next, end=end)
            else:
                Agent.update_Q_learning(R, State, action,
                                        para_next=para_next,
                                        adj=ADJ, end=end)
                METHOD = "Q-LEARNING"

            action_log[episode, n] = action
            R_log[episode, n] = R
            R_max_log[episode, n] = R_max
            R_min_log[episode, n] = R_min
            R_mean_log[episode, n] = R_mean

    print("Progress: 100%")

    # %% PLOT

    # Caculate the Line Of Sight angle for transmitter in GLOBAL coordinate system
    AoA_LOS_t = np.abs(np.arctan2(pos_log[0][1], pos_log[0][0]) * 180 / np.pi)

    # Caculate the Line Of Sight angle for receiver in GLOBAL coordinate system
    AoA_LOS_r_GLOBAL = np.arctan2(-pos_log[0][1], -pos_log[0][0])
    AoA_LOS_r_GLOBAL.shape = (N, 1)

    # Caculate the Line Of Sight angle for receiver in LOCAL coordinate system
    AoA_LOS_r_LOCAL = np.abs(helpers.get_local_angle(AoA_LOS_r_GLOBAL[-1][0], Orientation[-1][0][2, :]) * 180 / np.pi)

    beam_LOS = helpers.angle_to_beam(AoA_LOS_r_LOCAL, W)

    NN = 3000

    if NN > chunksize:
        NN = chunksize-1

    if NN > 50:
        MM = 50
    else:
        MM = NN

    ACC = np.sum(beam_LOS[-NN:, 0] == action_log[0, -NN:]) / NN
    print(f"METHOD: {METHOD}, ACC: {ACC}, AJD: {ADJ}, ORI: {ORI}")

    print("Starts plotting")
    plots.n_lastest_scatter(action_log[0, :], beam_LOS, MM, ["action", "beam"],
                            "Receiver - beam")

    plots.n_lastest_scatter_ylog(action_log[0, :], beam_LOS, NN, ["Max R", "Taken R"],
                                 "Reward plot", marker=".")

    plots.mean_reward(R_max_log, R_mean_log, R_min_log, R_log,
                      ["R_max", "R_mean", "R_min", "R"], "Mean Rewards")

    plots.positions(pos_log, r_lim)
    print("Done")
