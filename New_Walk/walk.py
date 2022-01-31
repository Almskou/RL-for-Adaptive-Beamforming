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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Position")
    plt.plot(pos[:, 0], pos[:, 1])
    plt.xlim([-2000, 2000])
    plt.ylim([-2000, 2000])
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def plot_velocity(v, T, delta_t, case):
    x_axis = np.linspace(0, T, int(np.floor(T/delta_t)))

    plt.figure()
    plt.title("Velocity")
    plt.plot(x_axis, v)
    plt.ylim([0, case["vmax"]])
    plt.show()


def plot_direction(phi, T, delta_t):
    x_axis = np.linspace(0, T, int(np.floor(T/delta_t)))

    plt.figure()
    plt.title("Direction")
    plt.plot(x_axis, phi)
    plt.show()


def plot_vel_dir(v, phi, T, delta_t, case):
    x_axis = np.linspace(0, T, int(np.floor(T/delta_t)))

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('dir[rad]', color=color)
    ax1.plot(x_axis, phi, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([-np.pi, np.pi])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('vel [m/s]', color=color)  # we already handled the x-label with ax1
    ax2.plot(x_axis, v, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, case["vmax"]])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
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

        self.radius_limit = 2000

    def set_acceleration(self, acc):
        if acc:
            return np.random.rand()*self.acc_max + 0.00001
        return - (np.random.rand()*self.dec_max + 0.00001)

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
                self.a = self.set_acceleration(True)
            elif self.v_target < v:
                self.a = self.set_acceleration(False)
            else:
                self.a = 0

        # Update the velocity bases on target and accelation
        v = v + self.a*delta_t

        if (((self.a > 0) and (v > self.v_target)) or
                ((self.a < 0) and (v < self.v_target))):
            v = self.v_target
            self.a = 0

        return v

    def update_direction(self, phi, delta_t, v):
        # "Stop-turn-and-go" implemented here
        if v == 0:
            # Only changes the target delta phi once
            if not self.v_stop:
                if np.random.rand() < self.pdirchange_stop:
                    if np.random.rand() < 0.5:
                        delta_phi_target = np.pi/2
                    else:
                        delta_phi_target = -np.pi/2
                else:
                    delta_phi_target = 0

                # Calculat the number of time step the change in direction needs
                self.curve_time = np.floor((np.random.rand()*(self.ctmax-self.ctmin) + self.ctmin)/delta_t)

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
                self.curve_time = np.floor((np.random.rand()*(self.ctmax-self.ctmin) + self.ctmin)/delta_t)

                # Resets the tracker
                self.curve_dt = 0

                # Target direction change
                delta_phi_target = (np.random.rand()*2*np.pi - np.pi)

                # Calculate the delta direction change per time step
                self.delta_phi = delta_phi_target/self.curve_time

                # Calculate the maximum radius
                rc = self.v_target*self.curve_time*delta_t/np.abs(delta_phi_target)

                # Calculate the maximum velocity which can be taken
                self.vrmax = np.sqrt(self.mu_s*9.81*rc)

                if self.v_target > self.vrmax:
                    self.v_target = self.vrmax

                if v > self.vrmax:
                    self.a = self.set_acceleration(False)

                    self.curve_slow = np.ceil(((v - self.vrmax)/np.abs(self.a))/delta_t)
                else:
                    self.curve_slow = 0

            # Updates the direction based on the target delta phi
            if self.curve_dt < self.curve_time + self.curve_slow:
                if self.curve_dt >= self.curve_slow:
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
                phi[t] = self.update_direction(phi[t-1], delta_t, v[t])
                t += 1

        return v, phi, pos


# %% Main
if __name__ == "__main__":
    hello_world()

    delta_t = 0.01
    T = 300

    case = load_case("car")
    track = track(case, delta_t)
    v, phi, pos = track.run(T, delta_t)

    plot_velocity(v, T, delta_t, case)
    plot_direction(phi, T, delta_t)
    plot_vel_dir(v, phi, T, delta_t, case)
    plot_walk(pos)
