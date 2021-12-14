# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
import matplotlib.pyplot as plt

import numpy as np

# %% Functions


def n_lastest_scatter(y1, y2, N, labels, title,
                      x1=None, x2=None, marker=None):
    if x1 is None:
        x1 = np.arange(N)

    if x2 is None:
        x2 = np.arange(N)

    plt.figure()
    plt.title(title)

    plt.scatter(x1, y1[-N:], label=labels[0], marker=marker)
    plt.scatter(x2, y2[-N:], label=labels[1], marker=marker)
    plt.legend()
    plt.show()


def n_lastest_scatter_ylog(y1, y2, N, labels, title,
                           x1=None, x2=None, marker=None):
    if x1 is None:
        x1 = np.arange(N)

    if x2 is None:
        x2 = np.arange(N)

    plt.figure()
    plt.title(title)

    plt.scatter(x1, y1[-N:], label=labels[0], marker=marker)
    plt.scatter(x2, y2[-N:], label=labels[1], marker=marker)
    plt.legend()
    plt.yscale('log')
    plt.show()


def mean_reward(y1, y2, y3, y4, labels, title,
                x1=None, x2=None, x3=None, x4=None, db=False):
    if x1 is None:
        x1 = np.arange(len(y1[0, :]))
    if x2 is None:
        x2 = np.arange(len(y2[0, :]))
    if x3 is None:
        x3 = np.arange(len(y3[0, :]))
    if x4 is None:
        x4 = np.arange(len(y4[0, :]))

    plt.figure()
    plt.title(title+f" - {len(y1)} Episodes")

    plt.plot(x1, np.mean(y1, axis=0), label=labels[0])
    plt.plot(x2, np.mean(y2, axis=0), label=labels[1])
    plt.plot(x3, np.mean(y3, axis=0), label=labels[2])
    plt.plot(x4, np.mean(y4, axis=0), label=labels[3])
    plt.legend()
    plt.xlabel("Number of Steps")
    plt.ylabel("Mean Reward")
    if db is not True:
        plt.yscale('log')
    plt.show()


def directivity(W, N, title):
    """
    Plots a directivity plot for a codebook
    :param W: Codebook
    :param N: Resolution of angles
    :param title: Title put on the figure
    :return: Nothing
    """
    # Calculate the directivity for a page in DFT-codebook
    beam = np.zeros((len(W), N))
    Theta = np.linspace(0, np.pi, N)
    # Sweep over range of angles, to calculate the normalized gain at each angle
    for i in range(N):
        # Hardcode the array steering vector for a ULA with len(W) elements
        A = (1 / np.sqrt(len(W[0, :]))) * np.exp(-1j * np.pi * np.cos(Theta[i]) * np.arange(0, len(W[0, :])))
        for j in range(len(W)):
            # The gain is found by multiplying the code-page with the steering vector
            beam[j, i] = np.abs(np.conjugate(W[j, :]).T @ A)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title(title)
    for j in range(len(W)):
        # Calculate the angle with max gain for each code-page.
        max_angle = np.pi - np.arccos(np.angle(W[j, 1]) / np.pi)

        # Plot the gain
        ax.plot(Theta, beam[j, :], label=f"{j}")
        ax.vlines(max_angle, 0, np.max(beam[j, :]),
                  colors='r', linestyles="dashed",
                  alpha=0.4)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=4)
    plt.show()


def positions(pos_log, r_lim):
    fig, ax = plt.subplots()
    ax.set_title("Kunst")
    ax.add_patch(plt.Circle((0, 0), r_lim, color='r', alpha=0.1))
    for m in range(len(pos_log)):
        ax.plot(pos_log[m][0, :], pos_log[m][1, :], label=f"{m}")

    ax.set_xlim([-r_lim, r_lim])
    ax.set_ylim([-r_lim, r_lim])
    if len(pos_log) < 10:
        plt.legend()
    plt.show()


def ori_lines(y1, y2, ori_discrete, labels, title, N1, N2,
              x1=None, x2=None):

    ori = ori_discrete[0][N1:N2]
    y1 = y1[:, N1:N2]
    y2 = y2[:, N1:N2]

    if x1 is None:
        x1 = np.arange(len(y1[0, :]))
    if x2 is None:
        x2 = np.arange(len(y2[0, :]))

    tmp = []
    for x in range(1, len(ori)):
        diff = np.abs(ori[x]-ori[x-1])
        if diff > 4:
            diff = 8 - diff

        if diff > 1:
            tmp.append(x)

    tmp = np.array(tmp)

    """
    tmp = [x for x in range(1, len(ori))
           if (np.abs(ori[x]-ori[x-1]) > 1)]
    """

    plt.figure()
    plt.title(title+f" - {len(y1)} Episodes")
    plt.plot(x1, np.mean(y1, axis=0), label=labels[0])
    plt.plot(x2, np.mean(y2, axis=0), label=labels[1])
    plt.legend()
    for t in tmp:
        plt.axvline(t, 0, 1, color='red')
    plt.xlabel("Number of Steps")
    plt.ylabel("Mean Reward")
    plt.yscale('log')
    plt.show()
