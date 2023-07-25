import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from F1TenthRacingDRL.DataTools.plotting_utils import *

def plot_algorithm_performance():
    a_df = pd.read_csv("Data/FinalExperiment_1/ExperimentData.csv")
    df = pd.read_csv("Data/FinalExperiment_1/CondensedExperimentData.csv")

    a_df = a_df[(a_df.TrainID == "TAL") & (a_df.TrainMap == "GBR")]
    a_df = a_df.sort_values(["Algorithm", "Architecture"])

    alg_df = df[(df.TrainID == "TAL") & (df.TrainMap == "GBR")]
    alg_df = alg_df.sort_values(["Algorithm", "Architecture"])

    xs = np.arange(3)
    width = 0.3
    br1 = xs - width
    br2 = xs 
    br3 = xs + width
    brs = [br1, br2, br3]

    plt.figure(figsize=(5, 2.5))
    x_data = alg_df[alg_df.Architecture == "Game"].Algorithm

    for i, a in enumerate(alg_df.Architecture.unique()):
        y_data = alg_df[alg_df.Architecture == a].Progress
        plt.bar(brs[i], y_data, label=a, width=width, color=color_pallet[i], alpha=0.3)
        for z in range(3):
            y_data_a = a_df[(a_df.Architecture == a) & (a_df.Repetition == z)].Progress 
            plt.plot(brs[i]-0.05 + 0.05*z, y_data_a, 'o', color=color_pallet[i], linewidth=2, markersize=8)

        
    plt.ylabel("Average Progress %")
    plt.grid(True)
    plt.xticks(xs, x_data)
    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(h, ["Full planning", "Trajectory tracking", "End-to-end"], ncol=3, bbox_to_anchor=(0.5, 1.0), loc='lower center')

    plt.tight_layout()
    plt.savefig("Data/FinalExperiment_1/Imgs/AlgorithmPerformance.svg", pad_inches=0.0, bbox_inches='tight')

def plot_algorithm_performance_times():
    a_df = pd.read_csv("Data/FinalExperiment_1/ExperimentData.csv").fillna(0)
    df = pd.read_csv("Data/FinalExperiment_1/CondensedExperimentData.csv")

    a_df = a_df[(a_df.TrainID == "TAL") & (a_df.TrainMap == "GBR")]
    a_df = a_df.sort_values(["Algorithm", "Architecture"])

    alg_df = df[(df.TrainID == "TAL") & (df.TrainMap == "GBR")]
    alg_df = alg_df.sort_values(["Algorithm", "Architecture"])

    xs = np.arange(3)
    width = 0.3
    br1 = xs - width
    br2 = xs 
    br3 = xs + width
    brs = [br1, br2, br3]

    plt.figure(figsize=(5, 2.5))
    plt.clf()
    x_data = alg_df[alg_df.Architecture == "Game"].Algorithm

    for i, a in enumerate(alg_df.Architecture.unique()):
        y_data = alg_df[alg_df.Architecture == a].Time
        plt.bar(brs[i], y_data, label=a, width=width, color=color_pallet[i], alpha=0.3)
        for z in range(3):
            y_data_a = a_df[(a_df.Architecture == a) & (a_df.Repetition == z)].Time
            plt.plot(brs[i]-0.05 + 0.05*z, y_data_a, 'o', color=color_pallet[i], linewidth=2, markersize=8)


        
    plt.ylabel("Lap time (s)")
    plt.grid(True)
    plt.xticks(xs, x_data)
    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(h, ["Full planning", "Trajectory tracking", "End-to-end"], ncol=3, bbox_to_anchor=(0.5, 1.0), loc='lower center')

    plt.tight_layout()
    plt.savefig("Data/FinalExperiment_1/Imgs/AlgorithmPerformanceTime.svg", pad_inches=0.0, bbox_inches='tight')

import matplotlib
def make_combined_plot():
    a_df = pd.read_csv("Data/FinalExperiment_1/ExperimentData.csv").fillna(0)
    df = pd.read_csv("Data/FinalExperiment_1/CondensedExperimentData.csv")

    a_df = a_df[(a_df.TrainID == "TAL") & (a_df.TrainMap == "GBR")]
    a_df = a_df.sort_values(["Algorithm", "Architecture"])

    alg_df = df[(df.TrainID == "TAL") & (df.TrainMap == "GBR")]
    alg_df = alg_df.sort_values(["Algorithm", "Architecture"])

    xs = np.arange(3)
    width = 0.3
    br1 = xs - width
    br2 = xs 
    br3 = xs + width
    brs = [br1, br2, br3]

    fig = plt.figure(figsize=(5.8, 2.2), constrained_layout=True)
    gs = matplotlib.gridspec.GridSpec(1, 2, figure=fig)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    x_data = alg_df[alg_df.Architecture == "Game"].Algorithm

    for i, a in enumerate(alg_df.Architecture.unique()):
        # y_data = alg_df[alg_df.Architecture == a].Success
        y_data = alg_df[alg_df.Architecture == a].Progress
        ax1.bar(brs[i], y_data, label=a, width=width, color=color_pallet[i], alpha=0.3)
        for z in range(3):
            y_data_a = a_df[(a_df.Architecture == a) & (a_df.Repetition == z)].Progress 
            ax1.plot(brs[i]-0.05 + 0.05*z, y_data_a, 'o', color=color_pallet[i], linewidth=2, markersize=5)
    ax1.set_xticks(xs, x_data)
    ax1.set_ylabel("Success rate (%)")

    for i, a in enumerate(alg_df.Architecture.unique()):
        y_data = alg_df[alg_df.Architecture == a].Time
        ax2.bar(brs[i], y_data, label=a, width=width, color=color_pallet[i], alpha=0.3)
        for z in range(3):
            y_data_a = a_df[(a_df.Architecture == a) & (a_df.Repetition == z)].Time
            ax2.plot(brs[i]-0.05 + 0.05*z, y_data_a, 'o', color=color_pallet[i], linewidth=2, markersize=5)

    ax2.set_ylabel("Lap time (s)")
    ax2.set_xticks(xs, x_data)
        
    ax1.grid(True, axis='y')
    ax2.grid(True, axis='y')
    h, l = plt.gca().get_legend_handles_labels()
    fig.legend(h, ["Full planning", "Trajectory tracking", "End-to-end"], ncol=3, bbox_to_anchor=(0.5, 0.95), loc='lower center')

    plt.tight_layout()
    plt.savefig("Data/FinalExperiment_1/Imgs/AlgorithmPerformance.svg", pad_inches=0.0, bbox_inches='tight')



make_combined_plot()


# plot_algorithm_performance()
# plot_algorithm_performance_times()