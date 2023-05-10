from RacingDRL.DataTools.plotting_utils import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

        
def make_laptime_and_success_barplot():
    base_path = "Data/"
    set_number = 2
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

    train_maps = ["gbr"]
    # train_maps = ["gbr", "mco"]

    fig, axs = plt.subplots(len(keys), len(folder_list), figsize=(8, 7), sharey=True)
    # fig, axs = plt.subplots(len(keys), len(folder_list), figsize=(5.5, 4.0), sharey=True)
    for k, key in enumerate(keys):
        for f, folder in enumerate(folder_list):
            for m, train_map in enumerate(train_maps):
                plt.sca(axs[k, f])
                
                mins, maxes, means = load_time_data(folder, f"{train_maps[m]}_TAL")
                # mins, maxes, means = load_time_data(folder, f"{train_maps[m]}_train")
                plt.bar(brs[m], means[key][0:4], color=pp_light[m+1], width=barWidth, label=train_map.upper())
                plot_error_bars(brs[m], mins[key][0:4], maxes[key], pp_darkest[m+1], w)
                
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.gca().set_xticks([0, 1, 2, 3], ["AUT", "ESP", "GBR", "MCO"])
            plt.grid(True)
            
            axs[0, f].set_title(f"{folder_labels[f]}")
        axs[k, 0].set_ylabel(ylabels[k])
        
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="center", bbox_to_anchor=(0.55, -0.01))
        
    name = f"Data/PerformanceTrainMaps_{set_number}"
    
    std_img_saving(name)
   
      
make_laptime_and_success_barplot()