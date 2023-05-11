from RacingDRL.DataTools.plotting_utils import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

        
def make_laptime_and_success_barplot():
    base_path = "Data/"
    # set_number = 8
    set_number = 5
    test_name = "PlanningMaps" 
    folder_games = base_path + test_name + f"_{set_number}/"
    
    test_name = "TrajectoryMaps" 
    # set_number = 8
    folder_traj = base_path + test_name + f"_{set_number}/"
    
    test_name = "EndMaps" 
    # set_number = 8
    folder_pp = base_path + test_name + f"_{set_number}/"
    
    folder_list = [folder_games, folder_traj, folder_pp]
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

    # train_maps = ["gbr"]
    train_maps = ["gbr", "mco"]

    fig, axs = plt.subplots(len(keys), len(train_maps), figsize=(5.5, 5.0), sharey=True)
    for m in range(len(train_maps)):
        for z in range(len(keys)):
            key = keys[z]
            # plt.sca(axs[z])
            plt.sca(axs[z, m])
            
            for f, folder in enumerate(folder_list):
                # mins, maxes, means = load_time_data(folder, f"{train_maps[m]}_train")
                mins, maxes, means = load_time_data(folder, f"{train_maps[m]}_TAL")
                plt.bar(brs[f], means[key][0:4], color=pp_light[f+1], width=barWidth, label=folder_labels[f])
                plot_error_bars(brs[f], mins[key][0:4], maxes[key], pp_darkest[f+1], w)
                
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.gca().set_xticks([0, 1, 2, 3], ["AUT", "ESP", "GBR", "MCO"])
            plt.grid(True)
            
            # axs[z].set_ylabel(ylabels[z])
            axs[z, 0].set_ylabel(ylabels[z])
        axs[0, m].set_title(f"{train_maps[m].upper()}")
        
    # handles, labels = axs[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, ncol=3, loc="center", bbox_to_anchor=(0.55, -0.01))
        
    name = f"Data/Imgs/PerformanceMaps_Barplot_{set_number}"
    
    std_img_saving(name)
   
      
make_laptime_and_success_barplot()