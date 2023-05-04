from RacingDRL.DataTools.plotting_utils import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def make_speed_barplot(folder, key, ylabel):
    plt.figure(figsize=(2.5, 1.9))
    # plt.figure(figsize=(3.3, 1.9))
    xs = np.arange(4, 9)
    
    barWidth = 0.4
    w = 0.05
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]

    mins, maxes, means = load_time_data(folder, "gbr")
    
    plt.bar(br1, means[key], color=pp_light[1], width=barWidth, label="GBR")
    plot_error_bars(br1, mins[key], maxes[key], pp_darkest[1], w)
    
    mins, maxes, means = load_time_data(folder, "mco")
    plt.bar(br2, means[key], color=pp_light[5], width=barWidth, label="MCO")
    plot_error_bars(br2, mins[key], maxes[key], pp_darkest[5], w)
        
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(1))
    plt.xlabel("Maximum speed (m/s)")
    plt.ylabel(ylabel)
    
    plt.legend()
        
    name = folder + f"SpeedBarPlot_{key}_{folder.split('/')[-2]}"
    
    std_img_saving(name)
   
def plot_speed_barplot_series(folder):
    keys = ["time", "success", "progress"]
    ylabels = "Time (s), Success (%), Progress (%)".split(", ")
    
    for i in range(len(keys)):
        make_speed_barplot(folder, keys[i], ylabels[i])
        
        
def make_laptime_and_success_barplot():
    base_path = "Data/"
    set_number = 8
    test_name = "PlanningMaps" 
    folder_games = base_path + test_name + f"_{set_number}/"
    
    test_name = "EndMaps" 
    set_number = 8
    folder_pp = base_path + test_name + f"_{set_number}/"
    
    test_name = "TrajectoryMaps" 
    set_number = 8
    folder_traj = base_path + test_name + f"_{set_number}/"
    
    xs = np.arange(4)
    
    barWidth = 0.26
    w = 0.05
    br1 = xs - barWidth
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth * 2 for x in br1]

    keys = ["time", "success", "progress"]
    ylabels = "Time (s), Success (%), Avg. Progress (%)".split(", ")
    keys = ["time", "progress"]
    ylabels = "Time (s), Avg. Progress (%)".split(", ")
    # keys = ["time", "success"]
    # ylabels = "Time (s), Success (%)".split(", ")

    train_maps = ["gbr", "mco"]

    fig, axs = plt.subplots(len(keys), len(train_maps), figsize=(5.5, 4.0), sharey=True)
    # fig, axs = plt.subplots(len(keys), len(train_maps), figsize=(10, 10), sharey=True)
    for m in range(len(train_maps)):
        for z in range(len(keys)):
            key = keys[z]
            
            plt.sca(axs[z, m])
            mins, maxes, means = load_time_data(folder_games, f"{train_maps[m]}_train")
            plt.bar(br1, means[key][0:4], color=pp_light[1], width=barWidth, label="Planning")
            plot_error_bars(br1, mins[key][0:4], maxes[key], pp_darkest[1], w)
            
            mins, maxes, means = load_time_data(folder_traj, f"{train_maps[m]}_train")
            plt.bar(br2, means[key][0:4], color=pp_light[2], width=barWidth, label="Trajectory")
            plot_error_bars(br2, mins[key][0:4], maxes[key], pp_darkest[2], w)
            
            mins, maxes, means = load_time_data(folder_pp, f"{train_maps[m]}_train")
            plt.bar(br3, means[key][0:4], color=pp_light[5], width=barWidth, label="Pure Pursuit")
            plot_error_bars(br3, mins[key][0:4], maxes[key], pp_darkest[5], w)
                
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            # plt.gca().set_ylabel(ylabels[z])
            plt.gca().set_xticks([0, 1, 2, 3], ["AUT", "ESP", "GBR", "MCO"])
            plt.grid(True)
            
            axs[z, 0].set_ylabel(ylabels[z])
        axs[0, m].set_title(f"{train_maps[m].upper()}")
        
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="center", bbox_to_anchor=(0.55, -0.01))
    # axs[0].set_xlabel("Maximum speed (m/s)")
    # axs[1].set_xlabel("Maximum speed (m/s)")
        
    name = f"Data/PerformanceMaps_Barplot"
    
    std_img_saving(name)
   
      
make_laptime_and_success_barplot()