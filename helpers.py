# -*- coding: utf-8 -*-
"""
@author: Dennis Sand, Nicolai Almskou,
         Peter Fisker & Victor Nissen
"""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt


# %% Functions
def steering_vectors2d(dir, theta, r, lambda_):
    e = dir * np.matrix([np.cos(theta), np.sin(theta)])
    return np.exp(-2j * (np.pi / lambda_) * e.T @ r)


def plot_directivity(W, N, title):
    # Calculating the directivity for a page in DFT-codebook
    beam = np.zeros((len(W), N))
    Theta = np.linspace(0, np.pi, N)
    # Sweep over range of angles, to calculate the angle with maximum "gain"
    for i in range(N):
        # Hardcode the array steering vector for a ULA with 10 elements
        A = np.exp(-1j * np.pi * np.cos(Theta[i]) * np.arange(0, len(W)))

        for j in range(len(W)):
            # The "gain" is found by multiplying the code-page with the steering vector
            beam[j, i] = np.abs(np.conjugate(W[j, :]).T @ A)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title(title)
    for j in range(len(W)):
        # Calculate the angle with max "gain".
        max_angle = np.pi - np.arccos(np.angle(W[j, 1]) / np.pi)

        # Plot the "gain"
        ax.plot(Theta, beam[j, :], label=f"{j}")
        ax.vlines(max_angle, 0, np.max(beam[j, :]),
                  colors='r', linestyles="dashed",
                  alpha=0.4)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=4)
    plt.show()
