from matplotlib import pyplot as plt
import numpy as np
from RacingDRL.DataTools.plotting_utils import *

def make_deviation_bar_plot():
    map_name = "mco"
    base_path = "Data/"
    set_number = 5
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]
    folder = base_path + f"LapWise_{set_number}/"
    
    folder_list = [folder + f"AgentOff_SAC_{vehicle_keys[i]}_{map_name}_TAL_8_5_0/" for i in range(3)]
    folder_list.append(folder + f"PurePursuit_PP_pathFollower_{map_name}_TAL_8_5_0/")
    

    inds = [4, 10, 6, 8]

    
    mean_lateral_list = []
    lateral_list_q1, lateral_list_q3 = [], []
    mean_speed_list = []
    speed_list_q1, speed_list_q3 = [], []

    for f, folder in enumerate(folder_list):
        read_file = folder + f"DetailSummaryStatistics{map_name.upper()}.txt"
        with open(read_file, 'r') as file:
            lines = file.readlines()
            line_data = lines[4]
            line_data = line_data.split(",")

        mean_lateral_list.append(float(line_data[6])*100)
        lateral_list_q1.append(float(line_data[7])*100)
        lateral_list_q3.append(float(line_data[8])*100)
        mean_speed_list.append(float(line_data[9]))
        speed_list_q1.append(float(line_data[10]))
        speed_list_q3.append(float(line_data[11]))

    mean_lateral_list = np.array(mean_lateral_list)
    lateral_list_q1 = np.array(lateral_list_q1)
    lateral_list_q3 = np.array(lateral_list_q3)
    mean_speed_list = np.array(mean_speed_list)
    speed_list_q1 = np.array(speed_list_q1)
    speed_list_q3 = np.array(speed_list_q3)

    xs = np.arange(4)
    fig, axs = plt.subplots(1, 2, figsize=(5, 1.8))
    # fig, axs = plt.subplots(1, 2, figsize=(5, 2))
    axs[0].bar(xs, mean_lateral_list, capsize=10, color=color_pallet, alpha=0.6)
    plt.sca(axs[0])
    plot_error_bars(xs, lateral_list_q1, lateral_list_q3, color_pallet, w=0.3, tails=False)
    # axs[0].bar(labels, mean_lateral_list, yerr=std_lateral_list, capsize=10, color=color_pallet)
    axs[0].set_ylabel("Lateral deviation (cm)")
    # axs[0].set_title("Lateral deviation (cm)", fontsize=10)
    axs[0].grid(True)
    # axs[0].set_xticks(xs, labels)
    plt.xticks([])
    
    axs[1].bar(xs, mean_speed_list,  capsize=10, color=color_pallet, alpha=0.6)
    plt.sca(axs[1])
    plot_error_bars(xs, speed_list_q1, speed_list_q3, color_pallet, w=0.3, tails=False)
    axs[1].set_ylabel("Speed deviation (m/s)", fontsize=10)
    # axs[1].set_title("Speed deviation (m/s)", fontsize=10)
    axs[1].grid(True)
    plt.xticks([])
    axs[0].yaxis.set_tick_params(labelsize=8)
    axs[1].yaxis.set_tick_params(labelsize=8)
    
    
    fig.legend(labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.95), fontsize=8)
    
    name = f"Data/LapWise_5/_Imgs/deviation_barplot_{set_number}_{map_name.upper()}"
    std_img_saving(name, True)
    
    

make_deviation_bar_plot()