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
DIST = True  # Include the dist in the state
FILENAME = "38_901_UMi_LOS_100000_4_02_03"  # After the "data_" or "data_pos_"
CASE = "walk"  # "walk" or "car"

# %% main
if __name__ == "__main__":

    # Load Scenario configuration
    with open(f'Cases/{CASE}.json', 'r') as fp:
        case = json.load(fp)

    # State parameters
    n_actions = 2
    n_ori = 2

    # Number of steps in a episode
    N = 100000

    # Chunk size (More "episodes" per episode)
    chunksize = 33000

    # Number of episodes
    M = 1

    # Radius for communication range [m]
    r_lim = case["rlim"]

    # Stepsize limits [m] [min, max]
    stepsize = [case["stepsize"]["min"],
                case["stepsize"]["max"]]

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
                                    [fc, N, M, r_lim, stepsize, scenarios])
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
            ori_discrete[m, :] = helpers.disrete_ori(Orientation[m][0][2, :], Nbr)
    else:
        ori_discrete = None

    if DIST:
        dist_discrete = np.zeros([M, N])
        for m in range(M):
            dist_discrete[m, :] = helpers.discrete_pos(pos_log[m], 4, r_lim)
    else:
        dist_discrete = None

    # Number of chuncks
    nchunk = int(np.floor(N/chunksize))

    # RUN
    action_log = np.zeros([M*nchunk, chunksize])
    R_log = np.zeros([M*nchunk, chunksize])
    R_max_log = np.zeros([M*nchunk, chunksize])
    R_min_log = np.zeros([M*nchunk, chunksize])
    R_mean_log = np.zeros([M*nchunk, chunksize])

    for m in range(M):
        for chunk in range(nchunk):
            print(f"Progress: {((m*nchunk + chunk) / (M*nchunk)) * 100:0.2f}%")
            # Create the Agent
            Agent = classes.Agent(action_space, eps=0.1)

            # Create the State
            if ORI:  # Orientation should be included in the state space
                if DIST:
                    State = classes.State([list(np.random.randint(0, Nbr, n_actions)),
                                           list([dist_discrete[0]]),
                                           list(np.random.randint(0, Nbr, n_ori))])
                else:
                    State = classes.State([list(np.random.randint(0, Nbr, n_actions)),
                                           list(["N/A"]),
                                           list(np.random.randint(0, Nbr, n_ori))])
            else:  # Only the actions should be included in the state space
                if DIST:
                    State = classes.State([list(np.random.randint(0, Nbr, n_actions)),
                                           list([dist_discrete[0]]),
                                           list(["N/A"])])
                else:
                    State = classes.State([list(np.random.randint(0, Nbr, n_actions)),
                                           list(["N/A"]),
                                           list(["N/A"])])

            # Update the enviroment data
            Env.update_data(AoA_Local[m][chunk*chunksize:(chunk+1)*chunksize],
                            AoD_Global[m][0][chunk*chunksize:(chunk+1)*chunksize],
                            coeff[m][0][chunk*chunksize:(chunk+1)*chunksize])

            # Initiate the action
            action = np.random.choice(action_space)

            end = False
            # Run the episode
            for n in range(chunksize):
                if ORI:
                    ori = int(ori_discrete[m, chunk*chunksize + n])
                    if n < chunksize-1:
                        next_ori = int(ori_discrete[m, chunk*chunksize + n + 1])
                else:
                    ori = None
                    next_ori = None

                if DIST:
                    dist = dist_discrete[m, chunk*chunksize + n]
                    if n < chunksize-1:
                        next_dist = dist_discrete[m, chunk*chunksize + n + 1]
                else:
                    dist = None
                    next_dist = None

                if n == chunksize-1:
                    end = True

                State.update_state(action, dist=dist, ori=ori)

                if ADJ:
                    action = Agent.e_greedy_adj(State.get_state(dist, ori), action)
                else:
                    action = Agent.e_greedy(State.get_state(dist, ori))

                R, R_max, R_min, R_mean = Env.take_action(n, action)

                if METHOD == "simple":
                    Agent.update(State, action, R, dist, ori)
                elif METHOD == "SARSA":
                    if ADJ:
                        next_action = Agent.e_greedy_adj(State.get_nextstate(action, dist=next_dist, ori=ori), action)
                    else:
                        next_action = Agent.e_greedy(State.get_nextstate(action, dist=next_dist, ori=ori))
                    Agent.update_sarsa(R, State, action,
                                       next_action, next_ori,
                                       next_dist, end=end)
                else:
                    Agent.update_Q_learning(R, State, action, next_ori,
                                            next_dist, adj=ADJ, end=end)
                    METHOD = "Q-LEARNING"

                action_log[m*nchunk+chunk, n] = action
                R_log[m*nchunk+chunk, n] = R
                R_max_log[m*nchunk+chunk, n] = R_max
                R_min_log[m*nchunk+chunk, n] = R_min
                R_mean_log[m*nchunk+chunk, n] = R_mean

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

    plots.mean_reward(R_mean_log, R_max_log, R_min_log, R_log,
                      ["R_mean", "R_max", "R_min", "R"], "Mean Rewards")

    plots.positions(pos_log, r_lim)
    print("Done")
