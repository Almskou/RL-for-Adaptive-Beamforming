# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 09:00:04 2022

@author: almsk
"""

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import json


# %% Functions
def hello_world():
    print("Hello World")


def create_track():
    pos = np.random.rand(2, 10)
    return pos


def plot_walk(pos):
    plt.figure()
    plt.title("Position")
    plt.plot(pos[:, 0], pos[:, 1])
    plt.show()


def plot_velocity(v, T, delta_t):
    x_axis = np.linspace(0, T, int(np.floor(T/delta_t)))

    plt.figure()
    plt.title("Velocity")
    plt.plot(x_axis, v)
    plt.show()


def plot_direction(phi, T, delta_t):
    x_axis = np.linspace(0, T, int(np.floor(T/delta_t)))

    plt.figure()
    plt.title("Direction")
    plt.plot(x_axis, phi)
    plt.show()


def load_case(CASE):
    # Load Scenario configuration
    with open(f'{CASE}.json', 'r') as fp:
        case = json.load(fp)
    return case


# %% class


class track():
    def __init__(self, case, delta_t):
        self.vpref = case["vpref"]
        self.vmax = case["vmax"]
        self.pvpref = case["pvpref"]
        self.pvuni = 1-np.sum(case["pvpref"])
        self.pvchange = delta_t/case["vchange"]
        self.pdirchange = delta_t/case["dirchange"]
        self.acc_max = case["acc_max"]
        self.dec_max = case["dec_max"]
        self.ctmax = case["curvetime"]["max"]
        self.ctmin = case["curvetime"]["min"]

        self.v_target = 0
        self.a = 0

        self.curve_time = 0
        self.curve_dt = 0
        self.delta_phi = 0

        self.radius_limit = 200

    def change_velocity(self):
        p_uni = np.random.rand()
        p_pref = self.pvpref[0]
        l_pref = len(self.pvpref)

        if p_uni < p_pref:
            return self.vpref[0]

        for i in range(1, l_pref):
            p_pref += self.pvpref[i]
            if (p_uni > p_pref - self.pvpref[i]) and (p_uni < p_pref):
                return self.vpref[i]

        return np.random.rand()*self.vmax

    def update_velocity(self, v, delta_t):
        if np.random.rand() < self.pvchange:
            self.v_target = self.change_velocity()

            # Get an accelation / deccelation
            if self.v_target > v:
                self.a = np.random.rand()*self.acc_max
            elif self.v_target < v:
                self.a = -np.random.rand()*self.dec_max
            else:
                self.a = 0

        # Update the velocity bases on target and accelation
        v = v + self.a*delta_t

        if (((self.a > 0) and (self.v_target < v)) or
                ((self.a < 0) and (self.v_target > v))):
            v = self.v_target

        return v

    def update_direction(self, phi, delta_t):
        if np.random.rand() < self.pdirchange:
            # Calculat the number of time step the change in direction needs
            self.curve_time = np.floor((np.random.rand()*(self.ctmax-self.ctmin) + self.ctmin)/delta_t)

            # Resets the tracker
            self.curve_dt = 0

            # Calculate the delta direction change per time step
            self.delta_phi = (np.random.rand()*2*np.pi - np.pi) / self.curve_time

        if self.curve_dt < self.curve_time:
            phi = phi + self.delta_phi

            # Checks for overflow
            if phi > np.pi:
                phi -= 2*np.pi
            if phi < -np.pi:
                phi += 2*np.pi

            self.curve_dt += 1

        return phi

    def update_pos(self, pos, v, phi, delta_t):
        # x-axis
        pos[0] = pos[0] + np.cos(phi)*v*delta_t

        # y-axis
        pos[1] = pos[1] + np.sin(phi)*v*delta_t

        return pos

    def run(self, T, delta_t):
        # Create a empty array for the velocities
        v = np.zeros([int(np.floor(T/delta_t)), 1])
        phi = np.zeros([int(np.floor(T/delta_t)), 1])
        pos = np.zeros([int(np.floor(T/delta_t)), 2])

        # Get start values
        self.v_target = self.change_velocity()
        v[0] = self.v_target
        phi[0] = np.random.rand()*2*np.pi - np.pi
        pos[0, :] = np.random.uniform(-self.radius_limit / 2, self.radius_limit / 2, size=2)

        # Start running the "simulation"
        t = 1
        i = 0
        while (t < int(np.floor(T/delta_t))):
            pos[t, :] = self.update_pos(pos[t-1, :], v[t-1], phi[t-1], delta_t)
            if np.linalg.norm(pos[t, :]) > self.radius_limit:
                # Restarts the run
                print(f'number of tries: {i}')
                print(f'How far we got: {t}')

                t = 1
                i += 1

                # Start with new values
                self.v_target = self.change_velocity()
                v[0] = self.v_target
                phi[0] = np.random.rand()*2*np.pi - np.pi

            else:
                v[t] = self.update_velocity(v[t-1], delta_t)
                phi[t] = self.update_direction(phi[t-1], delta_t)
                t += 1

        return v, phi, pos


# %% Main
if __name__ == "__main__":
    hello_world()

    delta_t = 0.01
    T = 300

    case = load_case("pedestrian")
    track = track(case, delta_t)
    v, phi, pos = track.run(T, delta_t)
    # plot_velocity(v, T, delta_t)
    # plot_direction(phi, T, delta_t)
    plot_walk(pos)
