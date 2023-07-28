import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from F1TenthRacingDRL.DataTools.plotting_utils import *

import matplotlib
def make_combined_plot():
    a_df1 = pd.read_csv("Data/FinalExperiment_1/ExperimentData.csv").fillna(0)
    df1 = pd.read_csv("Data/FinalExperiment_1/CondensedExperimentData.csv")
    a_df2 = pd.read_csv("Data/FinalExperiment_4/ExperimentData.csv").fillna(0)
    df2 = pd.read_csv("Data/FinalExperiment_4/CondensedExperimentData.csv")

    a_df1 = a_df1[a_df1.TrainMap == "GBR"]
    df1 = df1[df1.TrainMap == "GBR"]

    a_df2 = a_df2[a_df2.TrainMap == "MCO"]
    df2 = df2[df2.TrainMap == "MCO"]
    print(df2)

    df = pd.concat([df1, df2], axis=0)
    a_df = pd.concat([a_df1, a_df2], axis=0)

    algorithm = "TD3"
    algorithm = "SAC"
    a_df = a_df[(a_df.Algorithm == algorithm) & (a_df.TrainID == "TAL") & (a_df.TrainMap == a_df.TestMap) & (a_df.Repetition < 3)]
    a_df = a_df.sort_values(["Architecture", "TestMap"])

    df = df[(df.Algorithm == algorithm) & (df.TrainID == "TAL") & (df.TrainMap == df.TestMap)]
    df = df.sort_values(["Architecture", "TestMap"])
    print(df)
    # print(a_df)


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

    for i, a in enumerate(df.Architecture.unique()):
        y_data = df[df.Architecture == a].Progress
        ax1.bar(brs[i], y_data, label=a, width=width, color=color_pallet[i], alpha=0.3)
        for z in range(3):
            y_data_a = a_df[(a_df.Architecture == a) & (a_df.Repetition == z)].Progress 
            xs_pts = brs[i]-0.05 + 0.05*z
            ax1.plot(xs_pts, y_data_a, 'o', color=color_pallet[i], linewidth=2, markersize=5)

        y_data = df[df.Architecture == a].Success
        ax2.bar(brs[i], y_data, label=a, width=width, color=color_pallet[i], alpha=0.3)
        for z in range(3):
            y_data_a = a_df[(a_df.Architecture == a) & (a_df.Repetition == z)].Success 
            xs_pts = brs[i]-0.05 + 0.05*z
            ax2.plot(xs_pts, y_data_a, 'o', color=color_pallet[i], linewidth=2, markersize=5)

        y_data = df[df.Architecture == a].Time
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

    # x_data = ["CTH", "TAL"]
    x_data = df.TestMap.unique()
    # x_data = ["Cross-track \n& heading error", "Trajectory-aided \nlearning"]
    ax1.set_xticks(xs, x_data, fontsize=9)
    ax2.set_xticks(xs, x_data, fontsize=9)
    ax3.set_xticks(xs, x_data, fontsize=9)
        
    ax1.grid(True, axis='y')
    ax2.grid(True, axis='y')
    ax3.grid(True, axis='y')
    h, l = ax1.get_legend_handles_labels()
    print(l)
    leg = fig.legend(h, l, ncol=3, bbox_to_anchor=(0.5, 0.94), loc='lower center')
    # leg = fig.legend(h, ["Full planning", "Trajectory tracking", "End-to-end"], ncol=3, bbox_to_anchor=(0.5, 0.94), loc='lower center')
    leg.legend_handles[0]._alpha = 1
    leg.legend_handles[1]._alpha = 1
    leg.legend_handles[2]._alpha = 1
    # fig.legend(ncol=3, bbox_to_anchor=(0.5, 0.95), loc='lower center')

    plt.tight_layout()
    name = "Data/FinalExperiment_1/Imgs/MapPerformance"
    std_img_saving(name)


make_combined_plot()


# plot_algorithm_performance()
# plot_algorithm_performance_times()