# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""
# %% Imports
from collections import defaultdict
import numpy as np

import helpers


# %% Track
class Track():
    def __init__(self, case, delta_t, r_lim, debug_print=False):
        self.delta_t = delta_t
        self.env = case["environment"]
        self.vpref = case["vpref"]
        self.vmax = case["vmax"]
        self.vmin = case["vmin"]
        self.pvpref = case["pvpref"]
        self.pvuni = 1 - np.sum(case["pvpref"])
        self.pvchange = self.delta_t / case["vchange"]
        self.pdirchange = self.delta_t / case["dirchange"]
        self.pdirchange_stop = case["stop_dirchange"]
        self.mu_s = case["static_friction"]
        self.acc_max = case["acc_max"]
        self.dec_max = case["dec_max"]
        self.ctmax = case["curvetime"]["max"]
        self.ctmin = case["curvetime"]["min"]

        self.v_target = 0
        self.a = 0

        self.curve_time = 0
        self.curve_dt = 0
        self.delta_phi = 0
        self.v_stop = False
        self.vrmax = 0
        self.curve_slow = 0

        self.radius_limit = r_lim

        # Debug msg
        self.debug_print = debug_print

        # Base Stations Coordinates
        self.pos_bs = self.get_bs_pos()

    def get_bs_pos(self, olap=1):
        # Length to the points of the hexagon
        hex_point = self.radius_limit*olap

        # Length to the top of the hexagon
        hex_top = np.sqrt((hex_point**2) - (hex_point/2)**2)

        # a
        a = np.array([0, 0])

        # b
        b = np.array([a[0] - hex_point*1.5, hex_top])

        # d
        d = np.array([a[0] + hex_point*1.5, hex_top])

        # c
        c = np.array([a[0], 2*hex_top])

        return np.array([a, b, c, d]).T

    def set_acceleration(self, acc):
        if acc:
            return np.random.rand() * self.acc_max + 0.00001
        return - (np.random.rand() * self.dec_max + 0.00001)

    def change_velocity(self):
        p_uni = np.random.rand()
        p_pref = self.pvpref[0]
        l_pref = len(self.pvpref)

        # Checks if a pref. velocity should be chosen
        if p_uni < p_pref:
            return self.vpref[0]

        for i in range(1, l_pref):
            p_pref += self.pvpref[i]
            if (p_uni > p_pref - self.pvpref[i]) and (p_uni < p_pref):
                return self.vpref[i]

        # Return a velocity from a uniform dist. between set min and max
        return np.random.rand() * (self.vmax - self.vmin) + self.vmin

    def update_velocity(self, v):
        if np.random.rand() < self.pvchange:
            self.v_target = self.change_velocity()

            # Get an accelation / deccelation
            if self.v_target > v:
                self.a = self.set_acceleration(True)
            elif self.v_target < v:
                self.a = self.set_acceleration(False)
            else:
                self.a = 0

        # Update the velocity bases on target and accelation
        v = v + self.a * self.delta_t

        if (((self.a > 0) and (v > self.v_target)) or
                ((self.a < 0) and (v < self.v_target))):
            v = self.v_target
            self.a = 0

        return v

    def update_direction(self, phi, v):
        # "Stop-turn-and-go" implemented here
        if v == 0:
            # Only changes the target delta phi once
            if not self.v_stop:
                if np.random.rand() < self.pdirchange_stop:
                    if np.random.rand() < 0.5:
                        delta_phi_target = np.pi / 2
                    else:
                        delta_phi_target = -np.pi / 2
                else:
                    delta_phi_target = 0

                # Calculat the number of time step the change in direction needs
                self.curve_time = np.floor((np.random.rand() * (self.ctmax - self.ctmin) + self.ctmin) / self.delta_t)

                # Resets the tracker
                self.curve_dt = 0

                # Calculate the delta direction change per time step
                self.delta_phi = delta_phi_target / self.curve_time

                self.v_stop = True

        else:
            self.v_stop = False

            # Change target delta_phi, while the user is moving
            if np.random.rand() < self.pdirchange:
                # Calculat the number of time step the change in direction needs
                self.curve_time = np.floor((np.random.rand() * (self.ctmax - self.ctmin) + self.ctmin) / self.delta_t)

                # Resets the tracker
                self.curve_dt = 0

                # Target direction change
                delta_phi_target = (np.random.rand() * 2 * np.pi - np.pi)

                # Calculate the delta direction change per time step
                self.delta_phi = delta_phi_target / self.curve_time

                # Calculate the maximum radius
                rc = self.v_target * self.curve_time * self.delta_t / np.abs(delta_phi_target)

                # Calculate the maximum velocity which can be taken
                self.vrmax = np.sqrt(self.mu_s * 9.81 * rc)

                if self.v_target > self.vrmax:
                    self.v_target = self.vrmax

                if v > self.vrmax:
                    self.a = self.set_acceleration(False)

                    self.curve_slow = np.ceil(((v - self.vrmax) / np.abs(self.a)) / self.delta_t)
                else:
                    self.curve_slow = 0

            # Updates the direction based on the target delta phi
            if self.curve_dt < self.curve_time + self.curve_slow:
                if self.curve_dt >= self.curve_slow:
                    phi = phi + self.delta_phi

                    # Checks for overflow
                    phi = self.angle_overflow(phi)

                self.curve_dt += 1

        return phi

    def update_pos(self, pos, v, phi):
        # x-axis
        pos[0] = pos[0] + np.cos(phi) * v * self.delta_t

        # y-axis
        pos[1] = pos[1] + np.sin(phi) * v * self.delta_t

        return pos

    def angle_overflow(self, phi):
        # Checks for overflow
        if phi > np.pi:
            phi -= 2 * np.pi
        if phi < -np.pi:
            phi += 2 * np.pi

        return phi

    def initialise_run(self):
        # Velocity
        self.v_target = self.change_velocity()
        v = self.v_target

        # Position
        if self.env.lower() == "urban":
            c_idx = np.random.randint(0, 6)

            # Length to the points of the hexagon
            hex_point = self.radius_limit

            # Length to the top of the hexagon
            hex_top = np.sqrt((hex_point**2) - (hex_point/2)**2)

            # Can spawn around the 4 BS and in the middle of the BS.
            if c_idx < 4:
                pos = np.random.uniform(-self.radius_limit / 2, self.radius_limit / 2, size=2) + self.pos_bs[:, c_idx]
            elif c_idx == 5:
                pos = (np.random.uniform(-self.radius_limit*0.75, self.radius_limit*0.75, size=2) +
                       np.array([self.radius_limit / 2, hex_top]))
            else:
                pos = (np.random.uniform(-self.radius_limit*0.75, self.radius_limit*0.75, size=2) +
                       np.array([-self.radius_limit / 2, hex_top]))

        elif self.env.lower() == "highway":
            # Choose a start position on the edge based on a random chosen angle
            egde_angle = (np.random.rand() * 2 * np.pi - np.pi)
            pos = self.radius_limit * np.array([np.cos(egde_angle), np.sin(egde_angle)])

        else:
            pos = np.array([0, 0])

        # Direction
        if self.env.lower() == "urban":
            phi = np.random.rand() * 2 * np.pi - np.pi

        elif self.env.lower() == "highway":
            # Limit the start direction so it does not go out of the circle at the start

            # Get the angle which points at the center
            dir_center = egde_angle + np.pi

            # Checks for overflow
            dir_center = self.angle_overflow(dir_center)

            # Draw from a uniform distribution around the center angle
            edge_max = np.pi / 6
            edge_min = -np.pi / 6
            phi = dir_center + np.random.rand() * (edge_max - edge_min) + edge_min

            # Checks for overflow
            phi = self.angle_overflow(phi)

        else:
            phi = 0

        return v, phi, pos

    def run(self, N):
        # Create a empty array for the velocities
        v = np.zeros([N])
        phi = np.zeros([N])
        pos = np.zeros([3, N])
        pos[2, :] = 1.5

        # Get start values
        v[0], phi[0], pos[0:2, 0] = self.initialise_run()

        # Start running the "simulation"
        t = 1
        i = 0
        while (t < N):
            pos[0:2, t] = self.update_pos(pos[0:2, t - 1], v[t - 1], phi[t - 1])

            # Checks if the positions is inside the search radius of minimum one of the base stations
            if np.sum(np.linalg.norm(pos[0:2, t] - self.pos_bs.T, axis=1) < self.radius_limit) == 0:
                # Restarts the run
                if self.debug_print:
                    print(f'number of tries: {i}')
                    print(f'How far we got: {t}')

                t = 1
                i += 1

                # Start with new values
                v[0], phi[0], pos[0:2, 0] = self.initialise_run()

            else:
                v[t] = self.update_velocity(v[t - 1])
                phi[t] = self.update_direction(phi[t - 1], v[t])
                t += 1

        return pos


# %% Environment Class
class Environment():
    def __init__(self, W, F, Nt, Nr, Nbs,
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

        # Antenna Posistions
        self.r_t = r_t
        self.r_r = r_r

        # Wavelength
        self.lambda_ = 3e8 / fc

        # Transmitted power
        self.P_t = P_t

    def _get_reward(self, stepnr, action):

        # Empty Reward Matrix
        R = np.zeros((self.Nbs, len(self.F[:, 0]), len(self.W[:, 0])))

        # Calculate the reward pairs for each base station
        for b in range(self.Nbs):
            # Calculate steering vectors for transmitter and receiver
            alpha_rx = helpers.steering_vectors2d(direction=-1, theta=self.AoA[b, stepnr, :],
                                                  r=self.r_r, lambda_=self.lambda_)
            alpha_tx = helpers.steering_vectors2d(direction=1, theta=self.AoD[b, stepnr, :],
                                                  r=self.r_t, lambda_=self.lambda_)
            # Calculate channel matrix H
            H = np.zeros((self.Nr, self.Nt), dtype=np.complex128)
            for i in range(len(self.Beta[b, stepnr, :])):
                H += self.Beta[b, stepnr, i] * (alpha_rx[i].T @ np.conjugate(alpha_tx[i]))
            H = H * np.sqrt(self.Nr * self.Nt)

            # Calculate the reward
            for p in range(len(self.F[:, 0])):  # p - transmitter
                for q in range(len(self.W[:, 0])):  # q - receiver
                    R[b, p, q] = np.linalg.norm(np.sqrt(self.P_t) * np.conjugate(self.W[q, :]).T
                                                @ H @ self.F[p, :]) ** 2

        return np.max(R[:, :, action]), np.max(R), np.min(np.max(R, axis=1)), np.mean(np.max(R, axis=1))

    def take_action(self, stepnr, action):
        reward, max_reward, min_reward, mean_reward = self._get_reward(stepnr, action)

        return reward, max_reward, min_reward, mean_reward

    def update_data(self, AoA, AoD, Beta):
        self.AoA = AoA
        self.AoD = AoD
        self.Beta = Beta


# %% State Class
class State:
    def __init__(self, intial_state):
        self.state = intial_state

    def update_state(self, action, para=[None, None, None]):
        dist, ori, angle = para
        state_a = self.state[0][1:]
        state_a.append(action)

        if dist is not None:
            state_d = [dist]
        else:
            state_d = ["N/A"]

        if ori is not None:
            state_o = self.state[2][1:]
            state_o.append(ori)
        else:
            state_o = ["N/A"]

        if angle is not None:
            state_deg = [angle]
        else:
            state_deg = ["N/A"]

        self.state = [state_a, state_d, state_o, state_deg]

    def get_state(self, para=[None, None, None]):
        dist, ori, angle = para
        state_a = self.state[0]

        if dist is not None:
            state_d = self.state[1]
        else:
            state_d = ["N/A"]

        if ori is not None:
            state_o = self.state[2]
        else:
            state_o = ["N/A"]

        if angle is not None:
            state_deg = [angle]
        else:
            state_deg = ["N/A"]

        state = tuple([tuple(state_a), tuple(state_d),
                       tuple(state_o), tuple(state_deg)])

        return state

    def get_nextstate(self, action, para_next=[None, None, None]):
        dist, ori, angle = para_next
        next_state_a = self.state[0][1:]
        next_state_a.append(action)

        if dist is not None:
            next_state_d = [dist]
        else:
            next_state_d = ["N/A"]

        if ori is not None:
            next_state_o = self.state[2][1:]
            next_state_o.append(ori)
        else:
            next_state_o = ["N/A"]

        if angle is not None:
            next_state_deg = [angle]
        else:
            next_state_deg = ["N/A"]

        next_state = tuple([tuple(next_state_a), tuple(next_state_d),
                            tuple(next_state_o), tuple(next_state_deg)])
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
        self.Q = defaultdict(self._initiate_dict(0.001))
        self.accuracy = np.zeros(1)

    def _initiate_dict(self, value1, value2=0):
        """
        Small function used when initiating the dicts.
        For the alpha dict, value1 is alphas starting value.
        Value2 should be set to 0 as it is used to log the number of times it has been used.

        Parameters
        ----------
        value1 : FLOAT
            First value in the array.
        value2 : FLOAT, optional
            Second value in the array. The default is 0.

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
            if self.alpha[state, action][1] == 0:
                self.alpha[state, action] = [self.alpha_start * (1 / 1),
                                             1 + self.alpha[state, action][1]]
            else:
                self.alpha[state, action] = [self.alpha_start * (1 / self.alpha[state, action][1]),
                                             1 + self.alpha[state, action][1]]

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
        beam_dir = np.random.choice(self.action_space)
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
        actions = [self.action_space[(action - 1) % N],
                   self.action_space[action % N],
                   self.action_space[(action + 1) % N]]

        beam_dir = np.random.choice(actions)
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
            actions = [self.action_space[(action - 1) % N],
                       self.action_space[action % N],
                       self.action_space[(action + 1) % N]]
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
        beam_dir = np.random.choice(self.action_space)
        if self.Q[state, beam_dir][1] == 0:
            r_est = self.Q[state, beam_dir][0] + self.c * np.sqrt(np.log(t) / 1)
        else:
            r_est = self.Q[state, beam_dir][0] + self.c * np.sqrt(np.log(t) / self.Q[state, beam_dir][1])

        for action in self.action_space:
            if self.Q[state, action][1] == 0:
                r_est_new = self.Q[state, action][0] + self.c * np.sqrt(np.log(t) / 1)
            else:
                r_est_new = self.Q[state, action][0] + self.c * np.sqrt(np.log(t) / self.Q[state, action][1])
            if r_est_new > r_est:
                beam_dir = action
                r_est = r_est_new

        return beam_dir

    def update(self, State, action, reward, para):
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
        state = State.get_state(para)

        self.Q[state, action] = [(self.Q[state, action][0] +
                                  self.alpha[state, action][0] * (reward - self.Q[state, action][0])),
                                 self.Q[state, action][1] + 1]
        self._update_alpha(state, action)

    def update_sarsa(self, R, State, action, next_action, para_next, end=False):
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
        if end is False:
            next_state = State.get_nextstate(action, para_next)
            state = State.get_state(para_next)
            next_Q = self.Q[next_state, next_action][0]

            self.Q[state, action] = [self.Q[state, action][0] + self.alpha[state, action][0] *
                                     (R + self.gamma * next_Q - self.Q[state, action][0]),
                                     self.Q[state, action][1] + 1]
            self._update_alpha(state, action)
        else:
            state = State.get_state(para_next)
            next_Q = 0

            self.Q[state, action] = [self.Q[state, action][0] + self.alpha[state, action][0] *
                                     (R + self.gamma * next_Q - self.Q[state, action][0]),
                                     self.Q[state, action][1] + 1]
            self._update_alpha(state, action)

    def update_Q_learning(self, R, State, action, para_next, adj=False, end=False):
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
        if end is False:
            next_state = State.get_nextstate(action, para_next)
            state = State.get_state(para_next)
            if adj:
                next_action = self.greedy_adj(next_state, action)
            else:
                next_action = self.greedy(next_state)
            next_Q = self.Q[next_state, next_action][0]

            self.Q[state, action] = [self.Q[state, action][0] + self.alpha[state, action][0] *
                                     (R + self.gamma * next_Q - self.Q[state, action][0]),
                                     self.Q[state, action][1] + 1]
            self._update_alpha(state, action)
        else:
            state = State.get_state(para_next)
            next_Q = 0

            self.Q[state, action] = [self.Q[state, action][0] + self.alpha[state, action][0] *
                                     (R + self.gamma * next_Q - self.Q[state, action][0]),
                                     self.Q[state, action][1] + 1]
            self._update_alpha(state, action)
