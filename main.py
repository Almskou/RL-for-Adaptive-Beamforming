# -*- coding: utf-8 -*-
"""
@author: Nicolai Almskou & Victor Nissen
"""


# %% ---------- Imports ----------
import itertools
import numpy as np
import tensorflow as tf
from statistics import mean
from collections import namedtuple

import json
import argparse
from time import time

import helpers
from classes import Model, ReplayMemory, EpsilonGreedyStrategy, DQN_Agent, Environment

# Initialize tensorboard object
name = f'DQN_logs_{time()}'
summary_writer = tf.summary.create_file_writer(logdir=f'logs/{name}/')

# global parameters
RUN = False


# %% ---------- Functions ----------
def copy_weights(Copy_from, Copy_to):
    """
    Function to copy weights of a model to other
    """
    variables2 = Copy_from.trainable_variables
    variables1 = Copy_to.trainable_variables
    for v1, v2 in zip(variables1, variables2):
        v1.assign(v2.numpy())


def parser():
    description = """Adaptive Beamforming using Reinforcement Learning"""
    parser = argparse.ArgumentParser(description=description)

    help_str = """Name of the .json file which contains your test parameters.
                Default is the 'default.json' test parameters'"""
    parser.add_argument('--test_par', type=str,
                        default="test_env", help=help_str)

    return parser.parse_args()


# %% ---------- Main ----------
if __name__ == "__main__":

    # Initialize the parameters
    batch_size = 64
    gamma = 0.99
    eps_start = 1
    eps_end = 0.000
    eps_decay = 0.001
    target_update = 25
    memory_size = 100000
    lr = 0.01
    # epochs = 1000
    hidden_units = [200, 200]

    # debug: [plot, print, savefig]
    debug = [False, False, True]

    # Parse arguments
    args = parser()

    # Load test_parameters configuration
    with open(f'Test_parameters/{args.test_par}.json', 'r') as fp:
        settings = json.load(fp)

    # ----------- Channel Simulation Parameters -----------
    # Name of the data file afte "data_" and "data_pos_"
    FILENAME = settings["filename"]

    # Name of the chosen case "pedestrian", "car_urban" or "car_highway"
    CASE = settings["case"]

    # which engine should be be used "octave" OR "MATLAB"
    ENGINE = settings["engine"]

    # Possible scenarios for Quadriga simulations
    scenarios = settings["sim_par"]["scenarios"]

    # Number of steps in a episode
    N = settings["sim_par"]["N_steps"]

    # Sample Period [s]
    sample_period = settings["sim_par"]["sample_period"]

    # Number of episodes
    M = settings["sim_par"]["M_episodes"]

    # Multi user
    multi_user = settings["multi_user"]

    # Number of base stations
    Nbs = 4

    # Radius for communication range [m]
    r_lim = settings["sim_par"]["rlim"]

    # Intersite distance between base stations [m]
    intersite = settings["sim_par"]["intersite"]

    # ----------- Reinforcement Learning Parameters -----------
    # State parameters
    n_actions = 3
    n_ori = 3

    dist_res = 8
    angle_res = 8

    # Chunk size, number of samples taken out.
    chunksize = settings["test_par"]["chunk_size"]

    # Number of episodes per chunk
    epochs = settings["test_par"]["episodes"]

    # Which method RL should us: "simple", "SARSA" OR "Q-LEARNING"
    METHOD = settings["RL_par"]["method"]

    # ----------- Extracting variables from case -----------
    # Load Scenario configuration
    with open(f'Cases/{CASE}.json', 'r') as fp:
        case = json.load(fp)

    # Number of antennae
    Nt = case["transmitter"]["antennea"]  # Transmitter
    Nr = case["receiver"]["antennea"]  # Receiver

    # Number of beams
    Nbt = case["transmitter"]["beams"]  # Transmitter
    Nbr = case["receiver"]["beams"]  # Receiver

    fc = case["fc"]  # Center frequency
    lambda_ = 3e8 / fc  # Wave length
    P_t = case["P_t"]  # Transmission power

    # ----------- Create the data -----------
    # Take time on how long it takes to run the simulation / load data in
    t_start = time()
    # Load or create the data
    channel_par, pos_log = helpers.get_data(RUN, ENGINE, case, multi_user,
                                            f"data_pos_{FILENAME}.mat", f"data_{FILENAME}",
                                            [fc, N, M, r_lim, intersite, sample_period, scenarios, debug])

    print(f"Channel parameters generation took: {(time() - t_start):.3f} seconds", flush=True)

    # Take time on how long it take to the run the RL part
    t_start = time()

    # First entry are the BS coordinates
    if len(pos_log[0]) == 1:
        pos_bs = pos_log[0][0]
    else:
        pos_bs = pos_log[0]

    # Removes the BS from the pos_log
    pos_log = pos_log[1:]

    # Re-affirm that "M" matches data
    M = len(pos_log)

    # ----------- Extract data from Quadriga simulation -----------
    AoA_Global = channel_par[0][0]  # Angle of Arrival in Global coord. system
    AoD_Global = channel_par[1][0]  # Angle of Departure in Global coord. system
    coeff = channel_par[2][0]  # Channel Coefficients
    Orientation = channel_par[3][0]  # Orientation in Global coord. system

    if CASE == 'pedestrian':
        # Add some random noise to the orientation to simulate a moving person
        Orientation = helpers.noisy_ori(Orientation)

    # ----------- Prepare the simulation - Channel -----------
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

    # ----------- DQN -----------
    # Initialize the environment
    env = Environment(W, F, Nt, Nr, Nbs, Nbt, Nbr,
                      r_r, r_t, fc, P_t)

    """
    Notice that we are not using any function to make the states discrete here as DQN
    can handle discrete state spaces.
    """

    # Initialize Class variables
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = DQN_Agent(strategy, env.action_space_n)
    memory = ReplayMemory(memory_size)

    # Experience tuple variable to store the experience in a defined format
    Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'next_states', 'dones'])

    # Initialize the policy and target network
    policy_net = Model(len(env.state), hidden_units, env.action_space_n)
    target_net = Model(len(env.state), hidden_units, env.action_space_n)

    # Copy weights of policy network to target network
    copy_weights(policy_net, target_net)

    optimizer = tf.optimizers.Adam(lr)

    total_rewards = np.empty(epochs)

    for epoch in range(epochs):
        # Choose data for the episode
        data_idx = np.random.randint(0, N - chunksize) if (N - chunksize) else 0
        path_idx = np.random.randint(0, M)

        # Reset the environment with the new data
        state = env.reset(AoA_Local[path_idx][:, data_idx:data_idx + chunksize],
                          AoD_Global[path_idx][0][:, data_idx:data_idx + chunksize],
                          coeff[path_idx][0][:, data_idx:data_idx + chunksize])
        ep_rewards = 0
        losses = []

        for timestep in itertools.count():
            # Take action and observe next_stae, reward and done signal
            action, rate, flag = agent.select_action(state, policy_net)
            next_state, reward, done = env.step(action)
            ep_rewards += reward

            # Store the experience in Replay memory
            memory.push(Experience(state, action, next_state, reward, done))
            state = next_state

            if memory.can_provide_sample(batch_size):
                # Sample a random batch of experience from memory
                experiences = memory.sample(batch_size)
                batch = Experience(*zip(*experiences))

                # batch is a list of tuples, converting to numpy array here
                states = np.asarray(batch[0])
                actions = np.asarray(batch[1])
                rewards = np.asarray(batch[3])
                next_states = np.asarray(batch[2])
                dones = np.asarray(batch[4])

                # Calculate TD-target
                q_s_a_prime = np.max(target_net(np.atleast_2d(next_states).astype('float32')), axis=1)
                q_s_a_target = np.where(dones, rewards, rewards+gamma*q_s_a_prime)
                q_s_a_target = tf.convert_to_tensor(q_s_a_target, dtype='float32')

                # Calculate Loss function and gradient values for gradient descent
                with tf.GradientTape() as tape:
                    q_s_a = tf.math.reduce_sum(policy_net(np.atleast_2d(states).astype('float32'))
                                               * tf.one_hot(actions, env.action_space_n), axis=1)
                    loss = tf.math.reduce_mean(tf.square(q_s_a_target - q_s_a))

                # Update the policy network weights using ADAM
                variables = policy_net.trainable_variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))

                losses.append(loss.numpy())

            else:
                losses.append(0)

            # If it is time to update target network
            if timestep % target_update == 0:
                copy_weights(policy_net, target_net)

            if done:
                break

        total_rewards[epoch] = ep_rewards
        avg_rewards = total_rewards[max(0, epoch - 100):(epoch + 1)].mean()  # Running average reward of 100 iterations

        # Good old book-keeping
        with summary_writer.as_default():
            tf.summary.scalar('Episode_reward', total_rewards[epoch], step=epoch)
            tf.summary.scalar('Running_avg_reward', avg_rewards, step=epoch)
            tf.summary.scalar('Losses', mean(losses), step=epoch)

        if epoch % 1 == 0:
            print(f"Episode: {epoch} Episode_Reward: {total_rewards[epoch]} Avg_Reward: {avg_rewards: 0.1f}" +
                  f" Losses:{mean(losses): 0.1f} rate:{rate: 0.8f} flag:{flag}")
