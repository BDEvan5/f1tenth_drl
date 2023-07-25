import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from F1TenthRacingDRL.DataTools.plotting_utils import *

import matplotlib
def make_combined_plot():
    a_df = pd.read_csv("Data/FinalExperiment_1/ExperimentData.csv").fillna(0)
    df = pd.read_csv("Data/FinalExperiment_1/CondensedExperimentData.csv")

    a_df = a_df[(a_df.Algorithm == "SAC") & (a_df.TrainMap == "GBR")]
    a_df = a_df.sort_values(["TrainID", "Architecture"])
    a_df = a_df[a_df.TrainID != "Progress"]

    reward_df = df[(df.Algorithm == "SAC") & (df.TrainMap == "GBR")]
    reward_df = reward_df.sort_values(["TrainID", "Architecture"])
    reward_df = reward_df[reward_df.TrainID != "Progress"]

    xs = np.arange(2)
    width = 0.3
    br1 = xs - width
    br2 = xs 
    br3 = xs + width
    brs = [br1, br2, br3]

    fig = plt.figure(figsize=(5.8, 2.2), constrained_layout=True)
    gs = matplotlib.gridspec.GridSpec(1, 2, figure=fig)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    x_data = reward_df[reward_df.Architecture == "Game"].TrainID


    for i, a in enumerate(reward_df.Architecture.unique()):
        # y_data = reward_df[reward_df.Architecture == a].Success
        y_data = reward_df[reward_df.Architecture == a].Progress
        ax1.bar(brs[i], y_data, label=a, width=width, color=color_pallet[i], alpha=0.3)
        for z in range(3):
            # y_data_a = a_df[(a_df.Architecture == a) & (a_df.Repetition == z)].Success 
            y_data_a = a_df[(a_df.Architecture == a) & (a_df.Repetition == z)].Progress 
            xs_pts = brs[i]-0.05 + 0.05*z
            ax1.plot(xs_pts, y_data_a, 'o', color=color_pallet[i], linewidth=2, markersize=5)

        y_data = reward_df[reward_df.Architecture == a].Time
        ax2.bar(brs[i], y_data, width=width, color=color_pallet[i], alpha=0.3)
        for z in range(3):
            y_data_a = a_df[(a_df.Architecture == a) & (a_df.Repetition == z)].Time
            xs_pts = brs[i]-0.05 + 0.05*z
            ax2.plot(xs_pts, y_data_a, 'o', color=color_pallet[i], linewidth=2, markersize=5)

    ax1.set_xticks(xs, x_data)
    ax1.set_ylabel("Success rate (%)")
    ax2.set_ylabel("Lap time (s)")
    ax2.set_xticks(xs, x_data)
        
    ax1.grid(True, axis='y')
    ax2.grid(True, axis='y')
    h, l = ax1.get_legend_handles_labels()
    print(l)
    fig.legend(h, ["Full planning", "Trajectory tracking", "End-to-end"], ncol=3, bbox_to_anchor=(0.5, 0.94), loc='lower center')
    # fig.legend(ncol=3, bbox_to_anchor=(0.5, 0.95), loc='lower center')

    plt.tight_layout()
    plt.savefig("Data/FinalExperiment_1/Imgs/RewardPerformance.svg", pad_inches=0.0, bbox_inches='tight')



make_combined_plot()


# plot_algorithm_performance()
# plot_algorithm_performance_times()