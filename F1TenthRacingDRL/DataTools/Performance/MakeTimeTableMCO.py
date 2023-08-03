import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from F1TenthRacingDRL.DataTools.plotting_utils import *


def make_combined_plot_old():
    df1 = pd.read_csv("Data/FinalExperiment_1/CondensedExperimentData.csv")
    df2 = pd.read_csv("Data/FinalExperiment_4/CondensedExperimentData.csv")

    df1 = df1[df1.TrainMap == "GBR"]
    df2 = df2[df2.TrainMap == "MCO"]

    df = pd.concat([df1, df2], axis=0)

    algorithm = "TD3"
    # algorithm = "SAC"

    df = df[((df.Algorithm == algorithm) | (df.Algorithm == "PP")) & (df.TrainID == "TAL") & ((df.TrainMap == "MCO") | (df.TrainMap == "GBR"))]
    df = df.drop(["Algorithm", "TrainID", "full_name", "Distance", "MeanVelocity", "ProgressS"], axis=1)
    df = df.sort_values(["TestMap", "Architecture", "TrainMap"])

    df_gbr = df[df.TrainMap == "GBR"]
    df_gbr = df_gbr.drop(["TrainMap"], axis=1)
    df_mco = df[df.TrainMap == "MCO"]
    df_mco = df_mco.drop(["TrainMap"], axis=1)
    combined_df = pd.merge(df_gbr, df_mco, on=["TestMap", "Architecture"], suffixes=("_GBR", "_MCO"))
    combined_df = combined_df.sort_values(["TestMap", "Architecture"])
    # print(combined_df)
    combined_df.to_csv("Data/FinalExperiment_1/Imgs/MapTransfer.csv", index=False)

    df = df_mco.drop(["Progress", "Success"], axis=1)

    df.Architecture[df.Architecture == "Game"] = "Full planning"
    df.Architecture[df.Architecture == "TrajectoryFollower"] = "Trajectory tracking"
    df.Architecture[df.Architecture == "endToEnd"] = "End-to-end"
    df.Architecture[df.Architecture == "pathFollower"] = "Classic planner"

    df_vals = df.apply(lambda x: '{:.2f} $\pm$ {:.2f}'.format(x["Time"], x["TimeS"]), axis=1)

    df.insert(loc=1, column='TimeP', value=df_vals)
    df = df.drop(["Time", "TimeS"], axis=1)

    df = df.pivot(index="Architecture", columns="TestMap", values=["TimeP"])

    print(df)

    df.to_latex("Data/FinalExperiment_1/Imgs/TimeTransfer.tex", float_format="%.2f", index=True, escape=False)

def make_combined_plot():
    df = pd.read_csv("Data/Experiment_1/CondensedExperimentData.csv")

    algorithm = "TD3"
    # algorithm = "SAC"

    df = df[((df.Algorithm == algorithm) | (df.Algorithm == "PP")) & (df.TrainID == "TAL") & ((df.TrainMap == "MCO") | (df.TrainMap == "GBR"))]
    df = df.drop(["Algorithm", "TrainID", "full_name", "Distance", "MeanVelocity", "ProgressS"], axis=1)
    df = df.sort_values(["TestMap", "Architecture", "TrainMap"])

    df_gbr = df[df.TrainMap == "GBR"]
    df_gbr = df_gbr.drop(["TrainMap"], axis=1)
    df_mco = df[df.TrainMap == "MCO"]
    df_mco = df_mco.drop(["TrainMap"], axis=1)
    combined_df = pd.merge(df_gbr, df_mco, on=["TestMap", "Architecture"], suffixes=("_GBR", "_MCO"))
    combined_df = combined_df.sort_values(["TestMap", "Architecture"])
    # print(combined_df)
    combined_df.to_csv("Data/FinalExperiment_1/Imgs/MapTransfer.csv", index=False)

    df = df_mco.drop(["Progress", "Success"], axis=1)

    df.Architecture[df.Architecture == "Game"] = "Full planning"
    df.Architecture[df.Architecture == "TrajectoryFollower"] = "Trajectory tracking"
    df.Architecture[df.Architecture == "endToEnd"] = "End-to-end"
    df.Architecture[df.Architecture == "pathFollower"] = "Classic planner"

    df_vals = df.apply(lambda x: '{:.2f} $\pm$ {:.2f}'.format(x["Time"], x["TimeS"]), axis=1)

    df.insert(loc=1, column='TimeP', value=df_vals)
    df = df.drop(["Time", "TimeS"], axis=1)

    df = df.pivot(index="Architecture", columns="TestMap", values=["TimeP"])

    print(df)

    df.to_latex("Data/Experiment_1/Imgs/TimeTransfer.tex", float_format="%.2f", index=True, escape=False)

    

make_combined_plot()


# plot_algorithm_performance()
# plot_algorithm_performance_times()