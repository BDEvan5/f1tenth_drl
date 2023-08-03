import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from F1TenthRacingDRL.DataTools.plotting_utils import *

def make_combined_plot_old():
    a_df1 = pd.read_csv("Data/FinalExperiment_1/ExperimentData.csv").fillna(0)
    df1 = pd.read_csv("Data/FinalExperiment_1/CondensedExperimentData.csv")
    a_df2 = pd.read_csv("Data/FinalExperiment_4/ExperimentData.csv").fillna(0)
    df2 = pd.read_csv("Data/FinalExperiment_4/CondensedExperimentData.csv")

    a_df1 = a_df1[a_df1.TrainMap == "GBR"]
    df1 = df1[df1.TrainMap == "GBR"]

    a_df2 = a_df2[a_df2.TrainMap == "MCO"]
    df2 = df2[df2.TrainMap == "MCO"]

    df = pd.concat([df1, df2], axis=0)
    a_df = pd.concat([a_df1, a_df2], axis=0)

    algorithm = "TD3"
    # algorithm = "SAC"

    a_df = a_df[(a_df.Algorithm == algorithm) & (a_df.TrainID == "TAL") & ((a_df.TrainMap == "MCO") | (a_df.TrainMap == "GBR"))]
    a_df = a_df.drop(["Algorithm", "TrainID", "Distance", "MeanVelocity", "ProgressS"], axis=1)
    a_df = a_df.sort_values(["TestMap", "Architecture", "TrainMap"])

    a_df_gbr = a_df[a_df.TrainMap == "GBR"]
    a_df_gbr = a_df_gbr.drop(["TrainMap"], axis=1)
    a_df_mco = a_df[a_df.TrainMap == "MCO"]
    a_df_mco = a_df_mco.drop(["TrainMap"], axis=1)
    a_df = pd.merge(a_df_gbr, a_df_mco, on=["TestMap", "Architecture", "Repetition"], suffixes=("_GBR", "_MCO"))
    a_df = a_df.sort_values(["TestMap", "Architecture"])

    df = df[(df.Algorithm == algorithm) & (df.TrainID == "TAL") & ((df.TrainMap == "MCO") | (df.TrainMap == "GBR"))]
    df = df.drop(["Algorithm", "TrainID", "full_name", "Distance", "MeanVelocity", "ProgressS"], axis=1)
    df = df.sort_values(["TestMap", "Architecture", "TrainMap"])
    df_gbr = df[df.TrainMap == "GBR"]
    df_gbr = df_gbr.drop(["TrainMap"], axis=1)
    df_mco = df[df.TrainMap == "MCO"]
    df_mco = df_mco.drop(["TrainMap"], axis=1)
    df = pd.merge(df_gbr, df_mco, on=["TestMap", "Architecture"], suffixes=("_GBR", "_MCO"))
    df = df.sort_values(["TestMap", "Architecture"])
    print(df)

    xs = np.arange(4)
    width = 0.3
    br1 = xs - width
    br2 = xs 
    br3 = xs + width
    brs = [br1, br2, br3]

    fig = plt.figure(figsize=(5, 2))

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    for i, arch in enumerate(df.Architecture.unique()):
        # df_arch = df[df.Architecture == arch]
        # ax1.bar(brs[i], df_arch["Progress_GBR"], width, label=arch, color=color_pallet[i], alpha=0.4)
        # ax2.bar(brs[i], df_arch["Progress_MCO"], width, label=arch, color=color_pallet[i], alpha=0.4)
        # for z in range(3):
        #     m_df = a_df[a_df.Architecture == arch]
        #     m_df = m_df[m_df.Repetition == z]
        #     ax1.plot(brs[i], m_df["Progress_GBR"], 'o', width, color=color_pallet[i], alpha=0.99)
        #     ax2.plot(brs[i], m_df["Progress_MCO"], 'o', width, color=color_pallet[i], alpha=0.99)

        df_arch = df[df.Architecture == arch]
        ax1.bar(brs[i], df_arch["Success_GBR"], width, color=color_pallet[i], alpha=0.4)
        ax2.bar(brs[i], df_arch["Success_MCO"], width, label=arch, color=color_pallet[i], alpha=0.4)
        # for z in range(3):
        #     m_df = a_df[a_df.Architecture == arch]
        #     m_df = m_df[m_df.Repetition == z]
        #     ax1.plot(brs[i], m_df["Success_GBR"], 'o', width, color=color_pallet[i], alpha=0.99)
        #     ax2.plot(brs[i], m_df["Success_MCO"], 'o', width, color=color_pallet[i], alpha=0.99)

    ax1.set_title("Train Map: GBR")
    ax2.set_title("Train Map: MCO")
    ax1.set_xticks(xs)
    ax2.set_xticks(xs)
    ax1.set_xticklabels(df.TestMap.unique())
    ax2.set_xticklabels(df.TestMap.unique())

    ax1.grid(True, axis="y")
    ax2.grid(True, axis="y")
    fig.legend(["Full planning", "Trajectory tracking", "End-to-end"], ncol=3, bbox_to_anchor=(0.5, 0.9), loc='lower center')

    ax1.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax2.set_yticklabels([])
    ax1.set_ylabel("Completion \nrate (%)")

    name = "Data/FinalExperiment_1/Imgs/MapTransferCompletionRate"
    std_img_saving(name)

def make_combined_plot():
    a_df = pd.read_csv("Data/Experiment_1/ExperimentData.csv").fillna(0)
    df = pd.read_csv("Data/Experiment_1/CondensedExperimentData.csv")


    a_df = a_df[(a_df.TrainMap == "MCO") | (a_df.TrainMap == "GBR")]
    df = df[(df.TrainMap == "MCO") | (df.TrainMap == "GBR")]


    algorithm = "TD3"
    # algorithm = "SAC"

    a_df = a_df[(a_df.Algorithm == algorithm) & (a_df.TrainID == "TAL") & ((a_df.TrainMap == "MCO") | (a_df.TrainMap == "GBR"))]
    a_df = a_df.drop(["Algorithm", "TrainID", "Distance", "MeanVelocity", "ProgressS"], axis=1)
    a_df = a_df.sort_values(["TestMap", "Architecture", "TrainMap"])

    a_df_gbr = a_df[a_df.TrainMap == "GBR"]
    a_df_gbr = a_df_gbr.drop(["TrainMap"], axis=1)
    a_df_mco = a_df[a_df.TrainMap == "MCO"]
    a_df_mco = a_df_mco.drop(["TrainMap"], axis=1)
    a_df = pd.merge(a_df_gbr, a_df_mco, on=["TestMap", "Architecture", "Repetition"], suffixes=("_GBR", "_MCO"))
    a_df = a_df.sort_values(["TestMap", "Architecture"])

    df = df[(df.Algorithm == algorithm) & (df.TrainID == "TAL") & ((df.TrainMap == "MCO") | (df.TrainMap == "GBR"))]
    df = df.drop(["Algorithm", "TrainID", "full_name", "Distance", "MeanVelocity", "ProgressS"], axis=1)
    df = df.sort_values(["TestMap", "Architecture", "TrainMap"])
    df_gbr = df[df.TrainMap == "GBR"]
    df_gbr = df_gbr.drop(["TrainMap"], axis=1)
    df_mco = df[df.TrainMap == "MCO"]
    df_mco = df_mco.drop(["TrainMap"], axis=1)
    df = pd.merge(df_gbr, df_mco, on=["TestMap", "Architecture"], suffixes=("_GBR", "_MCO"))
    df = df.sort_values(["TestMap", "Architecture"])
    print(df)

    xs = np.arange(4)
    width = 0.3
    br1 = xs - width
    br2 = xs 
    br3 = xs + width
    brs = [br1, br2, br3]

    fig = plt.figure(figsize=(5, 2))

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    for i, arch in enumerate(df.Architecture.unique()):
        # df_arch = df[df.Architecture == arch]
        # ax1.bar(brs[i], df_arch["Progress_GBR"], width, label=arch, color=color_pallet[i], alpha=0.4)
        # ax2.bar(brs[i], df_arch["Progress_MCO"], width, label=arch, color=color_pallet[i], alpha=0.4)
        # for z in range(3):
        #     m_df = a_df[a_df.Architecture == arch]
        #     m_df = m_df[m_df.Repetition == z]
        #     ax1.plot(brs[i], m_df["Progress_GBR"], 'o', width, color=color_pallet[i], alpha=0.99)
        #     ax2.plot(brs[i], m_df["Progress_MCO"], 'o', width, color=color_pallet[i], alpha=0.99)

        df_arch = df[df.Architecture == arch]
        ax1.bar(brs[i], df_arch["Success_GBR"], width, color=color_pallet[i], alpha=0.4)
        ax2.bar(brs[i], df_arch["Success_MCO"], width, label=arch, color=color_pallet[i], alpha=0.4)
        # for z in range(3):
        #     m_df = a_df[a_df.Architecture == arch]
        #     m_df = m_df[m_df.Repetition == z]
        #     ax1.plot(brs[i], m_df["Success_GBR"], 'o', width, color=color_pallet[i], alpha=0.99)
        #     ax2.plot(brs[i], m_df["Success_MCO"], 'o', width, color=color_pallet[i], alpha=0.99)

    ax1.set_title("Train Map: GBR")
    ax2.set_title("Train Map: MCO")
    ax1.set_xticks(xs)
    ax2.set_xticks(xs)
    ax1.set_xticklabels(df.TestMap.unique())
    ax2.set_xticklabels(df.TestMap.unique())

    ax1.grid(True, axis="y")
    ax2.grid(True, axis="y")
    fig.legend(["Full planning", "Trajectory tracking", "End-to-end"], ncol=3, bbox_to_anchor=(0.5, 0.9), loc='lower center')

    ax1.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax2.set_yticklabels([])
    ax1.set_ylabel("Completion \nrate (%)")

    name = "Data/Experiment_1/Imgs/MapTransferCompletionRate"
    std_img_saving(name)



make_combined_plot()


# plot_algorithm_performance()
# plot_algorithm_performance_times()