# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 08:24:06 2022

@author: almsk
"""

import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

import helpers


class Model(tf.keras.Model):
    """
    Subclassing a multi-layered NN using Keras from Tensorflow
    """

    def __init__(self, num_states, hidden_units, num_actions):
        super(Model, self).__init__()  # Used to run the init method of the parent class
        self.input_layer = kl.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []

        for hidden_unit in hidden_units:
            self.hidden_layers.append(kl.Dense(hidden_unit, activation='tanh'))  # Left kernel initializer

        self.output_layer = kl.Dense(num_actions, activation='linear')

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output


class ReplayMemory():
    """
    Used to store the experience genrated by the agent over time
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy():
    """
    Decaying Epsilon-greedy strategy
    """

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1*current_step*self.decay)


class DQN_Agent():
    """
    Used to take actions by using the Model and given strategy.
    """

    def __init__(self, strategy, num_actions):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            return random.randrange(self.num_actions), rate, True
        else:
            return np.argmax(policy_net(np.atleast_2d(np.atleast_2d(state).astype('float32')))), rate, False


class Environment():
    """
    Environment created from the QuaDRiga simulation
    """

    def __init__(self, W, F, Nt, Nr, Nbs, Nbt, Nbr,
                 r_r, r_t, fc, P_t):
        self.AoA = 0
        self.AoD = 0
        self.Beta = 0

        # Codebooks
        self.W = W
        self.F = F

        # Number of antennas
        self.Nt = Nt
        self.Nr = Nr

        # Number of basestations
        self.Nbs = Nbs

        # Action space
        self.action_space_n = Nbr

        # Antenna Posistions
        self.r_t = r_t
        self.r_r = r_r

        # Wavelength
        self.lambda_ = 3e8 / fc

        # Transmitted power
        self.P_t = P_t

        # Number of steps / step count
        self.nstep = 0
        self.stepnr = 0

        # What state we are in
        self.state = np.random.choice(range(self.action_space_n), size=3)

    def _get_reward(self, action):

        # Empty Reward Matrix
        R = np.zeros((self.Nbs, len(self.F[:, 0]), len(self.W[:, 0])))

        # Calculate the reward pairs for each base station
        for b in range(self.Nbs):
            # Calculate steering vectors for transmitter and receiver
            alpha_rx = helpers.steering_vectors2d(direction=-1, theta=self.AoA[b, self.stepnr, :],
                                                  r=self.r_r, lambda_=self.lambda_)
            alpha_tx = helpers.steering_vectors2d(direction=1, theta=self.AoD[b, self.stepnr, :],
                                                  r=self.r_t, lambda_=self.lambda_)
            # Calculate channel matrix H
            H = np.zeros((self.Nr, self.Nt), dtype=np.complex128)
            for i in range(len(self.Beta[b, self.stepnr, :])):
                H += self.Beta[b, self.stepnr, i] * (alpha_rx[i].T @ np.conjugate(alpha_tx[i]))
            H = H * np.sqrt(self.Nr * self.Nt)

            # Calculate the reward
            for p in range(len(self.F[:, 0])):  # p - transmitter
                for q in range(len(self.W[:, 0])):  # q - receiver
                    R[b, p, q] = np.linalg.norm(np.sqrt(self.P_t) * np.conjugate(self.W[q, :]).T
                                                @ H @ self.F[p, :]) ** 2

        return np.max(R[:, :, action]), np.max(R), np.min(np.max(R, axis=1)), np.mean(np.max(R, axis=1))

    def _start_state(self):

        # Select a random start state
        self.state = np.random.choice(range(self.action_space_n), size=3)

        return self.state

    def _start_update(self, action):

        # Insert new action in the front of the array
        state_tmp = np.insert(self.state, 0, action)

        # Remove the last element so keep the same state dim.
        self.state = np.delete(state_tmp, -1)

        return self.state

    def step(self, action):

        # Get the reward
        reward, _, _, _ = self._get_reward(action)

        # Get the next_state
        next_state = self._state_update(action)

        # Update counter and see if episode if finished
        self.stepnr += 1
        if self.stepnr == self.nstep - 1:
            done = True
        else:
            done = False

        return next_state, reward, done

        # return reward, max_reward, min_reward, mean_reward

    def reset(self, AoA, AoD, Beta):
        # Reset step counter
        self.stepnr = 0

        # Get number of steps in the episode
        self.nstep = np.size(AoA)  # TODO: need to be checked

        # Save updated variables
        self.AoA = AoA
        self.AoD = AoD
        self.Beta = Beta

        # Return the start state()
        return self._start_state()
