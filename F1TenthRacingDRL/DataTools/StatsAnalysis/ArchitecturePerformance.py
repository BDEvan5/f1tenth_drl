from matplotlib import pyplot as plt
import numpy as np
from F1TenthRacingDRL.DataTools.plotting_utils import *
from matplotlib.ticker import MultipleLocator
import pandas as pd

def make_speed_performance_plot():
    base_path = "Data/FinalExperiment_2/"
    a_df = pd.read_csv(f"{base_path}ExperimentData.csv").fillna(0)

    a_df = a_df[ (a_df.TrainMap == "MCO")]
    a_df = a_df.sort_values(["Architecture"])
    a_df = a_df[a_df.TrainID == "TAL"]

    fig, axs = plt.subplots(1, 2, figsize=(5, 2))

    for i, arch in enumerate(a_df.Architecture.unique()):
        mini_df = a_df[a_df.Architecture == arch]
        xs = np.ones(len(mini_df)) * i 
        if len(mini_df) > 1:
            xs = xs + np.linspace(-0.3, 0.3, len(mini_df))

        axs[1].plot(xs, mini_df.Success, 'o', color=color_pallet[i], markersize=5)
        axs[1].bar(i, mini_df.Success.mean(), color=color_pallet[i], alpha=0.3)

        axs[0].plot(xs, mini_df.Time, 'o', color=color_pallet[i], markersize=5)
        axs[0].bar(i, mini_df.Time.mean(), color=color_pallet[i], alpha=0.3)


    axs[1].set_ylabel("Success rate (%)")
    axs[0].set_ylabel("Lap time (s)")

    axs[0].yaxis.set_major_locator(MultipleLocator(5))
    axs[1].yaxis.set_major_locator(MultipleLocator(20))
    axs[0].set_ylim(22, 52)

    axs[1].grid(True, axis='y')
    axs[0].grid(True, axis='y')
    plt.tight_layout()
    
    std_img_saving_path = base_path + f"Imgs/PerformanceSafetyPlot_MCO"
    std_img_saving(std_img_saving_path, True)


make_speed_performance_plot()