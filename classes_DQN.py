# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 08:24:06 2022

@author: almsk
"""


# %% ---------- Imports ----------
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


# %% ---------- Classes ----------
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
