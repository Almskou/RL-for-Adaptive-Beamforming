# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
import numpy as np
from collections import defaultdict

import helpers


# %% Track
class Track():

    def __init__(self, limit, stepsize):
        """
        Initializer for the class.
        :param limit: The max value for the radius of the circle that bounds the track
        :param stepsize: List containing the min and max values for step size.
        """
        self.radius_limit = limit
        self.pos = [0, 0, 1.5]
        self.stepsize = stepsize

    def get_stepsize(self):
        """
        Picks a step size as a random number between the defined limits
        :return: step size in meters
        """
        return np.random.uniform(self.stepsize[0], self.stepsize[1])

    def get_direction(self):
        """
        Picks a random direction
        :return:
        """
        return np.random.uniform(0, 2 * np.pi)

    def take_step(self):
        """
        Take a step.
        If the step takes you out of the defined circle,
        signal that the episode should stop.
        :return: Coordinates of current position after the step and whether to stop or not
        """
        angle = self.get_direction()
        pos_new = [0, 0, 1.5]
        stop = False

        pos_new[0] = self.pos[0] + self.get_stepsize() * np.cos(angle)
        pos_new[1] = self.pos[1] + self.get_stepsize() * np.sin(angle)

        if np.linalg.norm(pos_new) > self.radius_limit:
            stop = True
            return self.pos, stop

        self.pos = pos_new
        return pos_new, stop

    def run(self, N):
        """
        Runs one episode with N steps and logs the positions.
        :param N: Number of steps
        :return:
        """
        self.pos[0:2] = np.random.uniform(-self.radius_limit / 2, self.radius_limit / 2, size=2)

        pos_log = np.zeros([3, N + 1])
        pos_log[:, 0] = self.pos

        stop = False
        n = 0
        while (n < N) and (not stop):
            pos_log[:, n + 1], stop = self.take_step()
            n += 1

        return np.delete(pos_log, np.s_[n:], axis=1)


# %% Environment Class
class Environment():
    def __init__(self, W, F, Nt, Nr,
                 r_r, r_t, fc, P_t):
        self.AoA = 0
        self.AoD = 0
        self.Beta = 0
        self.W = W
        self.F = F
        self.Nt = Nt
        self.Nr = Nr
        self.r_t = r_t
        self.r_r = r_r
        self.lambda_ = 3e8/fc
        self.P_t = P_t

    def _get_reward(self, stepnr, action):
        # Calculate steering vectors for transmitter and receiver
        alpha_rx = helpers.steering_vectors2d(direction=-1, theta=self.AoA[stepnr, :],
                                              r=self.r_r, lambda_=self.lambda_)
        alpha_tx = helpers.steering_vectors2d(direction=1, theta=self.AoD[stepnr, :],
                                              r=self.r_t, lambda_=self.lambda_)

        # Calculate channel matrix H
        H = np.zeros((self.Nr, self.Nt), dtype=np.complex128)
        for i in range(len(self.Beta[stepnr, :])):
            H += self.Beta[stepnr, i] * (alpha_rx[i].T @ np.conjugate(alpha_tx[i]))
        H = H * np.sqrt(self.Nr * self.Nt)

        # Calculate the reward
        R = np.zeros([len(self.F[:, 0]), 1])
        for i in range(len(self.F[:, 0])):
            R[i] = np.linalg.norm(np.sqrt(self.P_t) * np.conjugate(self.W[action, :]).T
                                  @ H @ self.F[i, :]) ** 2

        return np.max(R)

    def take_action(self, stepnr, action):
        reward = self._get_reward(stepnr, action)

        return reward

    def update_data(self, AoA, AoD, Beta):
        self.AoA = AoA
        self.AoD = AoD
        self.Beta = Beta


# %% State Class
class State:
    def __init__(self, intial_state):
        self.state = intial_state

    def update_state(self, action, ori=None):
        if ori is None:
            state = self.state[1:]
            state.append(action)
            self.state = state
        else:
            state_a = self.state[0][1:]
            state_a.append(action)
            state_o = self.state[1][1:]
            state_o.append(ori)
            self.state = [state_a, state_o]

    def get_state(self, ori=None):
        if ori is None:
            state = tuple(self.state)
        else:
            state_a = self.state[0]
            state_o = self.state[1]
            state = tuple([tuple(state_a), tuple(state_o)])
        return state

    def get_nextstate(self, action, ori=None):
        if ori is None:
            next_state = self.state[1:]
            next_state.append(action)
            next_state = tuple(next_state)
        else:
            next_state_a = self.state[0][1:]
            next_state_a.append(action)
            next_state_o = self.state[1][1:]
            next_state_o.append(ori)
            next_state = tuple([tuple(next_state_a), tuple(next_state_o)])
        return next_state


# %% Agent Class
class Agent:
    def __init__(self, action_space, alpha=["constant", 0.7], eps=0.1, gamma=0.7, c=200):
        self.action_space = action_space  # Number of beam directions
        self.alpha_start = alpha[1]
        self.alpha_method = alpha[0]
        self.alpha = defaultdict(self._initiate_dict(alpha[1]))
        self.eps = eps
        self.gamma = gamma
        self.c = c
        self.Q = defaultdict(self._initiate_dict(0.00001))
        self.accuracy = np.zeros(1)

    def _initiate_dict(self, value1, value2=1):
        """
        Small function used when initiating the dicts.
        For the alpha dict, value1 is alphas starting value.
        Value2 should be set to 1 as it is used to log the number of times it has been used.

        Parameters
        ----------
        value1 : FLOAT
            First value in the array.
        value2 : FLOAT, optional
            Second value in the array. The default is 1.

        Returns
        -------
        TYPE
            An iterative type which defaultdict can use to set starting values.

        """
        return lambda: [value1, value2]

    def _update_alpha(self, state, action):
        """
        Updates the alpha values if method "1/n" has been chosen

        Parameters
        ----------
        state : ARRAY
            Current position (x,y).
        action : INT
            Current action taken.

        Returns
        -------
        None.

        """
        if self.alpha_method == "1/n":
            self.alpha[state, action] = [self.alpha_start*(1/self.alpha[state, action][1]),
                                         1+self.alpha[state, action][1]]

    def greedy(self, state):
        """
        Calculate which action is expected to be the most optimum.

        Parameters
        ----------
        state : ARRAY
            Which position on the grid you are standing on (x,y).

        Returns
        -------
        INT
            The chosen action.

        """
        beam_dir = self.action_space[0]
        r_est = self.Q[state, beam_dir][0]

        for action in self.action_space:
            if self.Q[state, action][0] > r_est:
                beam_dir = action
                r_est = self.Q[state, action][0]

        return beam_dir

    def e_greedy(self, state):
        """
        Return a random action in the action space based on the epsilon value.
        Else return the same value as the greedy function

        Parameters
        ----------
        state : ARRAY
            Which position on the grid you are standing on (x,y).

        Returns
        -------
        INT
            The chosen action.

        """
        if np.random.random() > self.eps:
            return self.greedy(state)
        else:
            return np.random.choice(self.action_space)

    def greedy_adj(self, state, action):
        N = len(self.action_space)
        actions = [self.action_space[(action-1) % N],
                   self.action_space[action % N],
                   self.action_space[(action+1) % N]]

        beam_dir = actions[1]
        r_est = self.Q[state, beam_dir][0]

        for action in actions:
            if self.Q[state, action][0] > r_est:
                beam_dir = action
                r_est = self.Q[state, action][0]

        return beam_dir

    def e_greedy_adj(self, state, action):
        if np.random.random() > self.eps:
            return self.greedy_adj(state, action)
        else:
            N = len(self.action_space)
            actions = [self.action_space[(action-1) % N],
                       self.action_space[action % N],
                       self.action_space[(action+1) % N]]
            return np.random.choice(actions)

    def UCB(self, state, t):
        """
        Uses the Upper Bound Confidence method as a policy. See eq. (2.10)
        in the book:
        Reinforcement Learning - An introduction.
        Second edition by Richard S. Sutton and Andrew G. Barto

        Parameters
        ----------
        state : ARRAY
            Current position (x,y).
        t : INT
            Current time step.

        Returns
        -------
        beam_dir : INT
            Beam direction.

        """
        beam_dir = self.action_space[0]
        r_est = self.Q[state, beam_dir][0] + self.c*np.sqrt(np.log(t)/self.Q[state, beam_dir][1])

        for action in self.action_space:
            r_est_new = self.Q[state, action][0] + self.c*np.sqrt(np.log(t)/self.Q[state, action][1])
            if r_est_new > r_est:
                beam_dir = action
                r_est = r_est_new

        return beam_dir

    def update(self, State, action, reward, ori):
        """
        Update the Q table for the given state and action based on equation (2.5)
        in the book:
        Reinforcement Learning - An introduction.
        Second edition by Richard S. Sutton and Andrew G. Barto

        Parameters
        ----------
        state : ARRAY
            Which position on the grid you are standing on (x,y).
        action : INT
            The action you are taking.
        reward : MATRIX
            The reward matrix.

        Returns
        -------
        None.

        """
        state = State.get_state(ori)

        self.Q[state, action] = [(self.Q[state, action][0] +
                                  self.alpha[state, action][0] * (reward - self.Q[state, action][0])),
                                 self.Q[state, action][1]+1]
        self._update_alpha(state, action)

    def update_sarsa(self, R, State, action, next_action, ori):
        """
        Update the Q table for the given state and action based on equation (6.7)
        in the book:
        Reinforcement Learning - An introduction.
        Second edition by Richard S. Sutton and Andrew G. Barto

        Parameters
        ----------
        R : MATRIX
            The reward matrix.
        state : ARRAY
            Which position on the grid you are standing on (x,y).
        action : INT
            The action you are taking.
        next_action : INT
            The next action you take.

        Returns
        -------
        None.

        """
        next_state = State.get_nextstate(action, ori)
        state = State.get_state(ori)
        next_Q = self.Q[next_state, next_action][0]

        self.Q[state, action] = [self.Q[state, action][0] + self.alpha[state, action][0] *
                                 (R + self.gamma * next_Q - self.Q[state, action][0]),
                                 self.Q[state, action][1]+1]
        self._update_alpha(state, action)

    def update_Q_learning(self, R, State, action, ori, adj=False):
        """
        Update the Q table for the given state and action based on equation (6.8)
        in the book:
        Reinforcement Learning - An introduction.
        Second edition by Richard S. Sutton and Andrew G. Barto

        Parameters
        ----------
        R : MATRIX
            The reward matrix.
        state : ARRAY
            Which position on the grid you are standing on (x,y).
        action : INT
            The action you are taking.

        Returns
        -------
        None.

        """
        next_state = State.get_nextstate(action, ori)
        state = State.get_state(ori)
        if adj:
            next_action = self.greedy_adj(next_state, action)
        else:
            next_action = self.greedy(next_state)
        next_Q = self.Q[next_state, next_action][0]

        self.Q[state, action] = [self.Q[state, action][0] + self.alpha[state, action][0] *
                                 (R + self.gamma * next_Q - self.Q[state, action][0]),
                                 self.Q[state, action][1]+1]
        self._update_alpha(state, action)
