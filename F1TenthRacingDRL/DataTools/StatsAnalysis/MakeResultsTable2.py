import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from F1TenthRacingDRL.DataTools.plotting_utils import *

def make_combined_plot():
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
    a_df = a_df[(a_df.Algorithm == algorithm) & (a_df.TrainID == "TAL") & (a_df.TrainMap == a_df.TestMap) & (a_df.Repetition < 3)]
    a_df = a_df.sort_values(["Architecture", "TestMap"])

    # df = df[(df.Algorithm == algorithm) & (df.TrainID == "TAL") & (df.TrainMap == "GBR")]
    df = df[(df.Algorithm == algorithm) & (df.TrainID == "TAL") & ((df.TrainMap == "MCO") | (df.TrainMap == "GBR"))]
    df = df.drop(["Algorithm", "TrainID", "full_name", "Distance", "MeanVelocity", "ProgressS"], axis=1)
    df = df.sort_values(["TestMap", "Architecture", "TrainMap"])

    df_gbr = df[df.TrainMap == "GBR"]
    df_gbr = df_gbr.drop(["TrainMap"], axis=1)
    df_mco = df[df.TrainMap == "MCO"]
    df_mco = df_mco.drop(["TrainMap"], axis=1)
    df = pd.merge(df_gbr, df_mco, on=["TestMap", "Architecture"], suffixes=("_GBR", "_MCO"))
    # df = pd.merge(df, on=["TestMap", "Architecture"], suffixes=("_GBR", "_MCO"))


    df = df.sort_values(["TestMap", "Architecture"])
    print(df)

    df.to_csv("Data/FinalExperiment_1/Imgs/MapTransfer.csv", index=False)

    plt.figure()
    
    xs = np.arange(4)
    width = 0.3
    br1 = xs - width
    br2 = xs 
    br3 = xs + width
    brs = [br1, br2, br3]

    fig = plt.figure()

    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    for i, arch in enumerate(df.Architecture.unique()):
        df_arch = df[df.Architecture == arch]
        ax1.bar(brs[i], df_arch["Success_GBR"], width, label=arch)
        ax2.bar(brs[i], df_arch["Success_MCO"], width, label=arch)
        # ax1.bar(brs[i], df_arch["Time_GBR"], width, label=arch)
        # ax2.bar(brs[i], df_arch["Time_MCO"], width, label=arch)

    ax1.set_title("GBR")
    ax2.set_title("MCO")
    ax1.set_xticks(xs)
    ax2.set_xticks(xs)
    ax1.set_xticklabels(df.TestMap.unique())
    ax2.set_xticklabels(df.TestMap.unique())

    plt.show()

make_combined_plot()


# plot_algorithm_performance()
# plot_algorithm_performance_times()