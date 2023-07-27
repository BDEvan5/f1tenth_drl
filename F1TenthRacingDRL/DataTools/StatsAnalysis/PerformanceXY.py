from matplotlib import pyplot as plt
import numpy as np
from F1TenthRacingDRL.DataTools.plotting_utils import *
from matplotlib.ticker import MultipleLocator
import pandas as pd

def make_speed_performance_plot():
    base_path = "Data/FinalExperiment_2/"
    a_df = pd.read_csv(f"{base_path}ExperimentData.csv").fillna(0)

    a_df = a_df[ (a_df.TrainMap == "MCO")]
    a_df = a_df.sort_values(["TrainID", "Architecture"])
    a_df = a_df[a_df.TrainID == "TAL"]
        
    plt.figure()
    for i, arch in enumerate(a_df.Architecture.unique()):
        color = color_pallet[i]
        mini_df = a_df[a_df.Architecture == arch]
        plt.plot(mini_df.Success, mini_df.Time, 'o', color=color, markersize=10, label=arch)

    plt.xlabel("Success rate (%)")
    plt.ylabel("Lap time (s)")

    plt.grid(True)
    plt.tight_layout()
    
    std_img_saving_path = base_path + f"Imgs/PerformanceSafetyPlot_MCO"
    std_img_saving(std_img_saving_path, True)


make_speed_performance_plot()