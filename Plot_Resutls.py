# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:22:52 2022

@author: almsk
"""

import json
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 5000
FONT_LEG = 16
FONT_LAB = 16


def smooth(scalars, weight: float):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def plot_loss(Basedir, Test, Test_No, Plotdir):
    with open(f'{Basedir}/{Test}/Test_{Test_No}_Loss.json', 'r') as fp:
        Loss = np.array(json.load(fp))[BATCH_SIZE:, 1:]

    print(f"Mean Loss - {Test} - Test No. {Test_No}: {np.mean(Loss[:, 1])}")

    Loss_Smooth = smooth(Loss[:, 1], 0.99999)

    plt.figure(figsize=(18, 6), dpi=160)
    # plt.title("Loss")
    plt.plot(Loss[:, 0], Loss[:, 1], label="Raw")
    plt.plot(Loss[:, 0], Loss_Smooth, label="Smooth")
    plt.xlabel("Step", fontsize=FONT_LAB)
    plt.ylabel("Loss", fontsize=FONT_LAB)
    plt.ylim(0, np.max(Loss_Smooth)+100)
    plt.legend(fontsize=FONT_LEG)
    plt.savefig(f"{Plotdir}/{Test}/Test_{Test_No}_Loss.pdf", bbox_inches='tight')
    plt.show()


def plot_mis(Basedir, Test, Test_No, Plotdir):
    with open(f'{Basedir}/{Test}/Test_{Test_No}_Mis-alignment-avg.json', 'r') as fp:
        Mis = np.array(json.load(fp))[BATCH_SIZE:, 1:]

    print(f"Mean Mis - {Test} - Test No. {Test_No}: {np.mean(Mis[:, 1])}")

    Mis_Smooth = smooth(Mis[:, 1], 0.99999)

    plt.figure(figsize=(18, 6), dpi=160)
    # plt.title("Mis-Alignment")
    plt.plot(Mis[:, 0], Mis[:, 1], label="Raw")
    plt.plot(Mis[:, 0], Mis_Smooth, label="Smooth")
    plt.xlabel("Step", fontsize=FONT_LAB)
    plt.ylabel("Mis-Alignment [dB]", fontsize=FONT_LAB)
    plt.legend(fontsize=FONT_LEG)
    plt.savefig(f"{Plotdir}/{Test}/Test_{Test_No}_Mis.pdf", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    Basedir = "Results"
    Plotdir = "Plots"
    Test = "Hidden_Layers"
    # Test_No = 1

    for Test_No in range(1, 8):
        # %% Plot Loss
        plot_loss(Basedir, Test, Test_No, Plotdir)

        # %% Plot Mis-Alignment
        plot_mis(Basedir, Test, Test_No, Plotdir)
