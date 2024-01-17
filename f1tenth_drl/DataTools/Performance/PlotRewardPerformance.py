import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from f1tenth_drl.DataTools.plotting_utils import *

import matplotlib

def make_combined_plot():
    # path = "Data/FinalExperiment_1/"
    path = "Data/Experiment_1/"
    a_df = pd.read_csv(f"{path}ExperimentData.csv").fillna(0)
    df = pd.read_csv(f"{path}CondensedExperimentData.csv").fillna(0)

    a_df = a_df[(a_df.Algorithm == "SAC") & (a_df.TrainMap == "GBR")]
    a_df = a_df.sort_values(["TrainID", "Architecture"])
    # a_df = a_df[a_df.TrainID != "Progress"]

    reward_df = df[(df.Algorithm == "SAC") & (df.TrainMap == "GBR")]
    reward_df = reward_df.sort_values(["TrainID", "Architecture"])
    # reward_df = reward_df[reward_df.TrainID != "Progress"]

    xs = np.arange(2)
    width = 0.3
    br1 = xs - width
    br2 = xs 
    br3 = xs + width
    brs = [br1, br2, br3]

    fig = plt.figure(figsize=(6., 2.1), tight_layout=True)
    gs = matplotlib.gridspec.GridSpec(1, 3, figure=fig)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    x_data = reward_df[reward_df.Architecture == "Game"].TrainID
    print(reward_df)
    print(a_df)

    for i, a in enumerate(reward_df.Architecture.unique()):
        y_data = reward_df[reward_df.Architecture == a].Progress
        ax1.bar(brs[i], y_data, label=a, width=width, color=color_pallet[i], alpha=0.3)
        for z in range(3):
            y_data_a = a_df[(a_df.Architecture == a) & (a_df.Repetition == z)].Progress
            # print(a)
            print(y_data_a)
            xs_pts = brs[i]-0.05 + 0.05*z
            ax1.plot(xs_pts, y_data_a, 'o', color=color_pallet[i], linewidth=2, markersize=5)

        y_data = reward_df[reward_df.Architecture == a].Success
        ax2.bar(brs[i], y_data, label=a, width=width, color=color_pallet[i], alpha=0.3)
        for z in range(3):
            y_data_a = a_df[(a_df.Architecture == a) & (a_df.Repetition == z)].Success 
            xs_pts = brs[i]-0.05 + 0.05*z
            ax2.plot(xs_pts, y_data_a, 'o', color=color_pallet[i], linewidth=2, markersize=5)

        y_data = reward_df[reward_df.Architecture == a].Time
        ax3.bar(brs[i], y_data, width=width, color=color_pallet[i], alpha=0.3)
        for z in range(3):
            y_data_a = a_df[(a_df.Architecture == a) & (a_df.Repetition == z)].Time
            xs_pts = brs[i]-0.05 + 0.05*z
            ax3.plot(xs_pts, y_data_a, 'o', color=color_pallet[i], linewidth=2, markersize=5)

    ax1.set_title("Average progress (%)", fontsize=10)
    ax2.set_title("Completion rate (%)", fontsize=10)
    ax3.set_title("Lap time (s)", fontsize=10)
    # ax1.set_ylabel("Average progress (%)")
    # ax2.set_ylabel("Completion rate (%)")
    # ax3.set_ylabel("Lap time (s)")
    
    ax1.yaxis.set_tick_params(labelsize=8)
    ax2.yaxis.set_tick_params(labelsize=8)
    ax3.yaxis.set_tick_params(labelsize=8)

    ax3.yaxis.set_major_locator(plt.MultipleLocator(10))

    x_data = ["CTH", "TAL"]
    # x_data = ["Cross-track \n& heading error", "Trajectory-aided \nlearning"]
    ax1.set_xticks(xs, x_data, fontsize=9)
    ax2.set_xticks(xs, x_data, fontsize=9)
    ax3.set_xticks(xs, x_data, fontsize=9)
        
    ax1.grid(True, axis='y')
    ax2.grid(True, axis='y')
    ax3.grid(True, axis='y')
    h, l = ax1.get_legend_handles_labels()
    print(l)
    leg = fig.legend(h, ["Full planning", "Trajectory tracking", "End-to-end"], ncol=3, bbox_to_anchor=(0.5, 0.94), loc='lower center')
    leg.legend_handles[0]._alpha = 1
    leg.legend_handles[1]._alpha = 1
    leg.legend_handles[2]._alpha = 1
    # fig.legend(ncol=3, bbox_to_anchor=(0.5, 0.95), loc='lower center')

    plt.tight_layout()
    name = f"{path}/Imgs/RewardPerformance"
    std_img_saving(name)


make_combined_plot()


# plot_algorithm_performance()
# plot_algorithm_performance_times()