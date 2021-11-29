# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:43:45 2021

@author: almsk
"""

import numpy as np
import matplotlib.pyplot as plt


class Track():
    def __init__(self, lim, stepsize):
        self.lim = lim
        self.pos = [0, 0, 1.5]
        self.stepsize = stepsize

    def get_stepsize(self):
        return np.random.uniform(self.stepsize[0], self.stepsize[1])

    def get_action(self):
        return np.random.rand()*2*np.pi

    def take_step(self):
        angle = self.get_action()
        pos_new = [0, 0, 1.5]

        pos_new[0] = self.pos[0] + self.get_stepsize()*np.cos(angle)
        pos_new[1] = self.pos[1] + self.get_stepsize()*np.sin(angle)

        if np.linalg.norm(pos_new) > self.lim:
            return self.pos, 1

        self.pos = pos_new
        return pos_new, 0

    def run(self, N):
        self.pos[0:2] = np.random.uniform(-self.lim/2, self.lim/2, size=2)

        pos_log = np.zeros([3, N+1])
        pos_log[:, 0] = self.pos

        stop = False
        n = 0
        while ((n < N) and (not stop)):
            pos = self.take_step()
            pos_log[:, n+1], stop = pos
            n += 1

        return np.delete(pos_log, np.s_[n:], axis=1)


if __name__ == "__main__":
    N = 10000
    track = Track(200, [0.5, 5])

    M = 5
    pos_log = []

    fig, ax = plt.subplots()
    ax.set_title("Kunst")
    ax.add_patch(plt.Circle((0, 0), 200, color='r', alpha=0.1))
    for m in range(M):
        pos_log.append(track.run(N))
        ax.plot(pos_log[m][0, :], pos_log[m][1, :])

    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
