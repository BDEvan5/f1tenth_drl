from RacingDRL.DataTools.plotting_utils import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

        
def make_laptime_and_success_barplot():
    set_number = 1
    base_path = f"Data/FinalExperiment_{set_number}/"
    folder_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    
    folder_labels = ["Full planning", "Trajectory tracking", "End-to-end"]
    
    xs = np.arange(4)
    barWidth = 0.26
    w = 0.05
    br1 = xs - barWidth
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth * 2 for x in br1]
    brs = [br1, br2, br3]

    keys = ["time", "success", "progress"]
    ylabels = "Time (s), Success (%), Avg. Progress (%)".split(", ")
    # keys = ["time", "progress"]
    # ylabels = "Time (s), Avg. Progress (%)".split(", ")
    # keys = ["time", "success"]
    # ylabels = "Time (s), Success (%)".split(", ")

    # map_name = "mco"
    map_name = "gbr"

    fig, axs = plt.subplots(1, len(keys), figsize=(5.5, 2.0))
    for z in range(len(keys)):
        key = keys[z]
        plt.sca(axs[z])
        # plt.sca(axs[z, m])
        
        for f, folder_key in enumerate(folder_keys):
            mins, maxes, means = load_time_data(base_path, f"{folder_key}_{map_name}_TAL")
            xs = brs[f][1:4]
            ys = means[key][1:4]
            plt.bar(xs, ys, color=color_pallet[f], width=barWidth, label=folder_labels[f], alpha=0.6)
            plot_error_bars(xs, mins[key][1:4], maxes[key][1:4], color_pallet[f], w, False)
            # plot_error_bars(xs, mins[key][0:4], maxes[key][0:4], color_pallet[f], w, False)
            
        plt.gca().xaxis.set_major_locator(MultipleLocator(1))
        plt.gca().set_xticks([1, 2, 3], ["ESP", "GBR", "MCO"])
        # plt.gca().set_xticks([0, 1, 2, 3], ["AUT", "ESP", "GBR", "MCO"])
        plt.grid(True)
        
        axs[z].set_title(ylabels[z])
        # axs[z].set_ylabel(ylabels[z])
        # axs[z, 0].set_ylabel(ylabels[z])
    # axs[0, m].set_title(f"{train_maps[m].upper()}")
    axs[1].get_yaxis().set_major_locator(MultipleLocator(25))
    axs[2].get_yaxis().set_major_locator(MultipleLocator(25))
        
    # handles, labels = axs[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, ncol=3, loc="center", bbox_to_anchor=(0.55, -0.01))
        
    name = f"{base_path}Imgs/ComparePerformance_train{map_name.upper()}_{set_number}"
    
    std_img_saving(name)
   
      
make_laptime_and_success_barplot()