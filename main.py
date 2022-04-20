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
import time
import sys
import os
import platform

import helpers
from classes import Model, ReplayMemory, EpsilonGreedyStrategy, DQN_Agent, Environment
import plots

# Initialize tensorboard object

# Get hostname
if platform.system() == "Windows":
    hostname = platform.uname().node
else:
    hostname = os.uname()[1]   # doesnt work on windows

local_time = time.strftime('%Y-%m-%d_%H_%M', time.localtime(time.time()))
name = f'DQN_logs_{local_time}_{hostname}'
summary_writer = tf.summary.create_file_writer(logdir=f'logs/{name}/main')
summary_writer_sub_1 = tf.summary.create_file_writer(logdir=f'logs/{name}/sub_1')
summary_writer_sub_2 = tf.summary.create_file_writer(logdir=f'logs/{name}/sub_2')
summary_writer_sub_3 = tf.summary.create_file_writer(logdir=f'logs/{name}/sub_3')

# global parameters
RUN = False
VALIDATE = False


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
                        default="default", help=help_str)

    help_str = """Call if the reinforcement learning should part should be run"""
    parser.add_argument('--DQN', action='store_true', help=help_str)

    return parser.parse_args()


# %% ---------- Main ----------
if __name__ == "__main__":
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

    # Batch size - number of memories used
    batch_size = settings["NN"]["Batch"]

    # Number of steps saved in the memory
    memory_size = settings["NN"]["Memory"]

    # How often the target policy should be updated. Every 'x' step
    target_update = settings["NN"]["Target"]

    # How many dense hidden layers and their size. [200, 100] - two layers of size 200 and 100
    hidden_units = settings["NN"]["hidden_layers"]

    # A factor given to the optimising algorithm
    lr = 0.01

    # Forgetting factor
    gamma = 0.99

    # Exploring probablity
    eps = settings["DQN"]["Epsilon"]
    eps_start = eps[0]  # Start prob.
    eps_end = eps[1]  # End prob.
    eps_decay = eps[2]  # how much it decays with until it hits the end pr. step.

    # Chunk size, number of samples taken out.
    chunksize = settings["test_par"]["chunk_size"]

    # Number of episodes per chunk
    epochs = settings["test_par"]["episodes"]

    # Which method RL should us: "simple", "SARSA" OR "Q-LEARNING"
    METHOD = settings["RL_par"]["method"]

    # Number of earlier actions in the state space
    n_earlier_actions = 3

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
    t_start = time.time()

    # Load or create the data
    channel_par, pos_log = helpers.get_data(RUN, ENGINE, case, multi_user,
                                            f"data_pos_{FILENAME}.mat", f"data_{FILENAME}",
                                            [fc, N, M, r_lim, intersite, sample_period, scenarios, debug])

    print(f"Channel parameters generation took: {(time.time() - t_start):.3f} seconds", flush=True)

    # Take time on how long it take to the run the RL part
    t_start = time.time()

    # First entry are the BS coordinates
    if len(pos_log[0]) == 1:
        pos_bs = pos_log[0][0]
    else:
        pos_bs = pos_log[0]

    # Removes the BS from the pos_log
    pos_log = pos_log[1:]

    # Re-affirm that "M" matches data
    M = len(pos_log)

    plots.positions(pos_log, pos_bs, r_lim, show=debug[0], save=debug[2])

    if not args.DQN:
        sys.exit("--DQN not called - stopping")

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
    env = Environment(W=W,
                      F=F,
                      Nt=Nt,
                      Nr=Nr,
                      Nbs=Nbs,
                      Nbt=Nbt,
                      Nbr=Nbr,
                      r_r=r_r,
                      r_t=r_t,
                      fc=fc,
                      P_t=P_t,
                      chunksize=chunksize,
                      AoA=AoA_Local,
                      AoD=AoD_Global,
                      Beta=coeff,
                      pos_log=pos_log,
                      n_earlier_actions=n_earlier_actions)

    env.create_reward_matrix()

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
    policy_net = Model(len(env.state), hidden_units, env.action_space_n, n_earlier_actions)
    target_net = Model(len(env.state), hidden_units, env.action_space_n, n_earlier_actions)

    # Copy weights of policy network to target network
    copy_weights(policy_net, target_net)

    optimizer = tf.optimizers.Adam(lr)

    total_rewards = np.empty(epochs)

    print("Prep work done")

    step = 1

    # Buffer for saving the x last y-db mis-alignment prob.
    mis_prob_buffer = np.array([[], [], [], []])

    # Load validation set
    if VALIDATE:
        idx_matrix = np.load(f"Validation_idx/validation_idx_{chunksize}.npy")

    for epoch in range(epochs):
        # Choose data for the episode
        if VALIDATE:
            data_idx = idx_matrix[0, epoch]
            path_idx = idx_matrix[1, epoch]
        else:
            data_idx = np.random.randint(0, N - chunksize) if (N - chunksize) else 0
            path_idx = np.random.randint(0, M)

        # Reset the environment with the new data
        state = env.reset(data_idx, path_idx)
        ep_rewards = 0
        losses = []

        for timestep in itertools.count():
            # Take action and observe next_stae, reward and done signal
            action, rate, flag = agent.select_action(state, policy_net)
            next_state, reward, done, max_reward, min_reward, mean_reward = env.step(action)
            ep_rewards += reward

            # Calculate the miss alignment prob. and add to the buffer
            mis_prob_buffer = np.insert(mis_prob_buffer, 0,
                                        [reward < max_reward - 3,
                                         reward < max_reward - 5,
                                         reward < max_reward - 7,
                                         reward < max_reward - 9], axis=1)

            # Ensure that the buffer only contain the latest 1000 steps
            if np.size(mis_prob_buffer, axis=1) > 1000:
                mis_prob_buffer = np.delete(mis_prob_buffer, -1, axis=1)

            mis_prob = np.mean(mis_prob_buffer, axis=1)

            # Log the reward
            with summary_writer.as_default():
                tf.summary.scalar('Step_reward', reward, step=step, description="Taken reward")
                tf.summary.scalar('Mis-alignment', mis_prob[0], step=step, description="3 dB")
            with summary_writer_sub_1.as_default():
                tf.summary.scalar('Step_reward', max_reward, step=step, description="Max reward")
                tf.summary.scalar('Mis-alignment', mis_prob[1], step=step, description="5 dB")
            with summary_writer_sub_2.as_default():
                tf.summary.scalar('Step_reward', min_reward, step=step, description="Mean reward")
                tf.summary.scalar('Mis-alignment', mis_prob[2], step=step, description="7 dB")
            with summary_writer_sub_3.as_default():
                tf.summary.scalar('Step_reward', mean_reward, step=step, description="Min reward")
                tf.summary.scalar('Mis-alignment', mis_prob[3], step=step, description="9 dB")

            step += 1

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
                loss = loss.numpy()

            else:
                losses.append(0)
                loss = 0

            with summary_writer.as_default():
                tf.summary.scalar('Loss', loss, step=step)

            # If it is time to update target network
            if timestep % target_update == 0:
                copy_weights(policy_net, target_net)

            if done:
                break

        total_rewards[epoch] = ep_rewards
        avg_rewards = total_rewards[max(0, epoch - 100):(epoch + 1)].mean()  # Running average reward of 100 iterations

        # Good old book-keeping
        # with summary_writer.as_default():
            # tf.summary.scalar('Episode_reward', total_rewards[epoch], step=epoch)
            # tf.summary.scalar('Running_avg_reward', avg_rewards, step=epoch)
            # tf.summary.scalar('Losses', mean(losses), step=epoch)

        if debug[1]:
            if epoch % 1 == 0:
                print(f"Episode: {epoch} Episode_Reward: {total_rewards[epoch]} Avg_Reward: {avg_rewards: 0.1f}" +
                      f" Losses:{mean(losses): 0.1f} rate:{rate: 0.8f} flag:{flag}")
