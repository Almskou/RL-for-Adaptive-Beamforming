# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:22:52 2022

@author: almsk
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['agg.path.chunksize'] = 10000

BATCH_SIZE = 5000
FONT_LEG = 16
FONT_LAB = 16
FIG_DPI = 160

FILETPYE = ".png"


def smooth(scalars, weight: float):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def plot_loss_json(Basedir, Test, Test_No, Plotdir, show_plots):
    with open(f'{Basedir}/{Test}/Test_{Test_No}_Loss.json', 'r') as fp:
        Loss = np.array(json.load(fp))[BATCH_SIZE:, 1:]

    print(f"Mean Loss - {Test} - Test No. {Test_No}: {np.mean(Loss[:, 1]):.3f}")

    Loss_Smooth = smooth(Loss[:, 1], 0.99999)

    plt.figure(figsize=(18, 6), dpi=FIG_DPI)
    plt.plot(Loss[:, 0], Loss[:, 1], label="Raw")
    plt.plot(Loss[:, 0], Loss_Smooth, label="Smooth")
    plt.xlabel("Step", fontsize=FONT_LAB)
    plt.ylabel("Loss", fontsize=FONT_LAB)
    plt.ylim(0, 150)
    plt.legend(fontsize=FONT_LEG)
    plt.savefig(f"{Plotdir}/{Test}/Test_{Test_No}_Loss{FILETPYE}", bbox_inches='tight')
    if show_plots:
        plt.show()


def plot_mis_json(Basedir, Test, Test_No, Plotdir, show_plots):
    with open(f'{Basedir}/{Test}/Test_{Test_No}_Mis-alignment-avg.json', 'r') as fp:
        Mis = np.array(json.load(fp))[BATCH_SIZE:, 1:]

    print(f"Mean Mis  - {Test} - Test No. {Test_No}: {np.mean(Mis[:, 1]):.3f}")

    Mis_Smooth = smooth(Mis[:, 1], 0.99999)

    plt.figure(figsize=(18, 6), dpi=FIG_DPI)
    plt.plot(Mis[:, 0], Mis[:, 1], label="Raw")
    plt.plot(Mis[:, 0], Mis_Smooth, label="Smooth")
    plt.xlabel("Step", fontsize=FONT_LAB)
    plt.ylabel("Misalignment [dB]", fontsize=FONT_LAB)
    plt.legend(fontsize=FONT_LEG)
    plt.ylim(0, 35)
    plt.savefig(f"{Plotdir}/{Test}/Test_{Test_No}_Mis{FILETPYE}", bbox_inches='tight')
    if show_plots:
        plt.show()


def plot_loss(Basedir, Test, Test_No, Plotdir, show_plots):
    Loss = np.load(f'{Basedir}/{Test}/Test_{Test_No}_loss.npy')[BATCH_SIZE:]

    print(f"Mean Loss - {Test} - Test No. {Test_No}: {np.mean(Loss):.3f}")

    Loss_Smooth = smooth(Loss, 0.99999)

    plt.figure(figsize=(18, 6), dpi=FIG_DPI)
    plt.plot(np.arange(0, len(Loss)), Loss, label="Raw")
    plt.plot(np.arange(0, len(Loss)), Loss_Smooth, label="Smooth")
    plt.xlabel("Step", fontsize=FONT_LAB)
    plt.ylabel("Loss", fontsize=FONT_LAB)
    plt.ylim(0, 150)
    plt.legend(fontsize=FONT_LEG)
    plt.savefig(f"{Plotdir}/{Test}/Test_{Test_No}_Loss{FILETPYE}", bbox_inches='tight')
    if show_plots:
        plt.show()


def plot_mis(Basedir, Test, Test_No, Plotdir, show_plots):
    Mis = np.load(f'{Basedir}/{Test}/Test_{Test_No}_avg_mis_alignment.npy')[BATCH_SIZE:]

    print(f"Mean Mis  - {Test} - Test No. {Test_No}: {np.mean(Mis):.3f}")

    Mis_Smooth = smooth(Mis, 0.99999)

    plt.figure(figsize=(18, 6), dpi=FIG_DPI)
    plt.plot(np.arange(0, len(Mis)), Mis, label="Raw")
    plt.plot(np.arange(0, len(Mis)), Mis_Smooth, label="Smooth")
    plt.xlabel("Step", fontsize=FONT_LAB)
    plt.ylabel("Misalignment [dB]", fontsize=FONT_LAB)
    plt.legend(fontsize=FONT_LEG)
    plt.ylim(0, 35)
    plt.savefig(f"{Plotdir}/{Test}/Test_{Test_No}_Mis{FILETPYE}", bbox_inches='tight')
    if show_plots:
        plt.show()


def plot_ECDF(Basedir, Test, Tests, Plotdir, show_plots, legends=None):

    fig = plt.figure(figsize=(18, 6), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    for Test_No in Tests:
        ECDF = np.load(f'{Basedir}/{Test}/Test_{Test_No}_mis_alignment.npy')[BATCH_SIZE:]
        sns.ecdfplot(ECDF[BATCH_SIZE:], legend=True)

    if legends is None:
        plt.legend(["Test " + f"{x}" for x in Tests], fontsize=FONT_LEG)
    else:
        plt.legend(["Test " + f"{x} - {legends[x-np.min(Tests)]}" for x in Tests], fontsize=FONT_LEG)
    plt.xlabel("Misalignment [dB]", fontsize=FONT_LAB)
    plt.ylabel("Proportion", fontsize=FONT_LAB)
    loc = mpl.ticker.MultipleLocator(base=0.1)  # Control the tick interval
    ax.yaxis.set_major_locator(loc)
    plt.grid()
    plt.xlim(-2, 70)  # 70
    plt.savefig(f"{Plotdir}/{Test}/ECDF_Mis.pdf", bbox_inches='tight')
    if show_plots:
        plt.show()


if __name__ == '__main__':
    Basedir = "../Results"
    Plotdir = "../Plots"
    Test = "Hidden_Layers"

    show_plots = False

    Captions = {"Hidden_Layers":    {"json":    True,
                                     "No_Tests":   7,
                                     "Loss":    True
                                     },
                "Hidden_Layers_2":  {"json":    True,
                                     "No_Tests":   7,
                                     "Loss":    True
                                     },
                "Embedding_Layer":  {"json":    True,
                                     "No_Tests":   7,
                                     "Loss":    True
                                     },
                "Batch_Memory":     {"json":    True,
                                     "No_Tests":   7,
                                     "Loss":    True
                                     },
                "Target":           {"json":    True,
                                     "No_Tests":   7,
                                     "Loss":    True
                                     },
                "Learning_Rate":    {"json":    True,
                                     "No_Tests":   7,
                                     "Loss":    True
                                     },
                "DQN":              {"json":    True,
                                     "No_Tests":   4,
                                     "Loss":    True
                                     },
                "Gamma":            {"json":    True,
                                     "No_Tests":   7,
                                     "Loss":    True
                                     },
                "Epsilon":          {"json":    True,
                                     "No_Tests":   7,
                                     "Loss":    True
                                     },
                "State":            {"json":    False,
                                     "No_Tests":   10,
                                     "Loss":    True
                                     },
                "State_2":          {"json":    False,
                                     "No_Tests":   6,
                                     "Loss":    True
                                     },
                "Reference":        {"json":    False,
                                     "No_Tests":    10,
                                     "Loss":    False
                                     },
                "Reference_end":    {"json":    False,
                                     "No_Tests":    4,
                                     "Loss":    False
                                     },
                "DQN_end":          {"json":    False,
                                     "No_Tests":    4,
                                     "Loss":    True
                                     },
                "State_NLOS":       {"json":    False,
                                     "No_Tests":    10,
                                     "Loss":    True
                                     },
                "DQN_16_32":        {"json":    False,
                                     "No_Tests":    4,
                                     "Loss":    True
                                     },
                "Reference_16_32":  {"json":    False,
                                     "No_Tests":    4,
                                     "Loss":    False
                                     },
                "DQN_32_64":        {"json":    False,
                                     "No_Tests":    4,
                                     "Loss":    True
                                     },
                "Reference_32_64":  {"json":    False,
                                     "No_Tests":    4,
                                     "Loss":    False
                                     },
                "Gamma_extra":      {"json":    False,
                                     "No_Tests":    3,
                                     "Loss":    True
                                     }
                }

    for Test in Captions:
        Tests = np.arange(1, Captions[Test]["No_Tests"]+1)
        # Tests = np.arange(7, 11)
        # Tests = [1, 2, 3]
        # Tests = [4, 5, 6]

        # legends = None
        legends = ["Car LOS", "Car NLOS", "Ped. LOS", "Ped. NLOS"]
        # legends = ["[3, 0, 0]", "[0, 3, 0]", "[0, 0, 3]"]
        # legends = ["[3, 3, 0]", "[0, 3, 3]", "[3, 0, 3]"]
        # legends = ["[1, 1, 1]", "[2, 2, 2]", "[3, 3, 3]", "[4, 4, 4]"]
        # legends = ["[1, 0, 0]", "[0, 1, 0]", "[0, 0, 1]"]
        # legends = ["[1, 1, 0]", "[0, 1, 1]", "[1, 0, 1]"]

        if Test == "Reference_32_64":
            for Test_No in Tests:
                if Captions[Test]["json"]:
                    if Captions[Test]["Loss"]:
                        plot_loss_json(Basedir, Test, Test_No, Plotdir, show_plots)

                    plot_mis_json(Basedir, Test, Test_No, Plotdir, show_plots)
                else:
                    if Captions[Test]["Loss"]:
                        plot_loss(Basedir, Test, Test_No, Plotdir, show_plots)

                    plot_mis(Basedir, Test, Test_No, Plotdir, show_plots)

            plot_ECDF(Basedir, Test, Tests, Plotdir, True, legends=legends)
