# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
import numpy as np


# %% Classes
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
