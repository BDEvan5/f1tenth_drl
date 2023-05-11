
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from RacingDRL.Utils.utils import *
from RacingDRL.DataTools.plotting_utils import *


def compare_training():
    base_path = "Data/"
    set_number = 5
    folder_keys = ["PlanningMaps", "TrajectoryMaps", "EndMaps"]
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end"]
    
    map_list = ["mco", "gbr"]
    
    max_speed = 8
    general_id = "TAL"
    n_repeats = 3
    n_train_steps = 60

    fig, axs = plt.subplots(1, 2, figsize=(5, 2.1))
    
    for m, map_name in enumerate(map_list):
        steps_list = []
        lap_time_list = []
        for a, architecture in enumerate(folder_keys):
            p = base_path + architecture + f"_{set_number}/"
            steps_list.append([])
            lap_time_list.append([])
            for j in range(n_repeats):
                path = p + f"AgentOff_SAC_{vehicle_keys[a]}_{map_name}_{general_id}_{max_speed}_{set_number}_{j}/"
                rewards, lengths, progresses, _ = load_csv_data(path)
                steps = np.cumsum(lengths) / 1000
                
                lap_times, completed_steps = [], []
                for z in range(len(progresses)):
                    if progresses[z] > 99:
                        lap_times.append(lengths[z]/10)
                        completed_steps.append(steps[z])
                        
                # avg_reward = true_moving_average(rewards[:-1], 30)
                steps_list[a].append(completed_steps)
                lap_time_list[a].append(lap_times)

        plt.sca(axs[m])
        xs = np.linspace(0, n_train_steps, 300)
        for i in range(len(steps_list)):
            colour_i = i + 0
            for j in range(n_repeats):
                plt.plot(steps_list[i][j], lap_time_list[i][j], '.-', color=color_pallet[colour_i], markersize=4, label=labels[i], alpha=0.5, linewidth=1)
                # plt.plot(steps_list[i][j], lap_time_list[i][j], '-', color=color_pallet[colour_i], linewidth=1, label=labels[i])
                
        plt.xlabel("Training Steps (x1000)")
        plt.title(f"{map_name.upper()}", size=10)
        # plt.ylim(0, 100)
        plt.grid(True)

    axs[0].set_ylabel("Lap times (s)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h[::3], l[::3], loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3)
    
    name = f"Data/Imgs/TrainingLapTimesComparison_{set_number}"
    std_img_saving(name)


def compare_training():
    base_path = "Data/"
    set_number = 5
    folder_keys = ["PlanningMaps", "TrajectoryMaps", "EndMaps"]
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end"]
    
    # map_list = ["mco", "gbr"]
    map_list = ["mco"]
    
    max_speed = 8
    general_id = "TAL"
    n_repeats = 3
    n_train_steps = 60

    fig, axs = plt.subplots(1, 1, figsize=(5, 2.1))
    
    for m, map_name in enumerate(map_list):
        steps_list = []
        lap_time_list = []
        for a, architecture in enumerate(folder_keys):
            p = base_path + architecture + f"_{set_number}/"
            steps_list.append([])
            lap_time_list.append([])
            for j in range(n_repeats):
                path = p + f"AgentOff_SAC_{vehicle_keys[a]}_{map_name}_{general_id}_{max_speed}_{set_number}_{j}/"
                rewards, lengths, progresses, _ = load_csv_data(path)
                steps = np.cumsum(lengths) / 1000
                
                lap_times, completed_steps = [], []
                for z in range(len(progresses)):
                    if progresses[z] > 99:
                        lap_times.append(lengths[z]/10)
                        completed_steps.append(steps[z])
                        
                # avg_reward = true_moving_average(rewards[:-1], 30)
                steps_list[a].append(completed_steps)
                lap_time_list[a].append(lap_times)

        # plt.sca(axs[m])
        for i in range(len(steps_list)):
            colour_i = i + 0
            for j in range(1):
            # for j in range(n_repeats):
                plt.plot(steps_list[i][j], lap_time_list[i][j], '.-', color=color_pallet[colour_i], markersize=3, label=labels[i], alpha=0.6, linewidth=2)
                # plt.plot(steps_list[i][j], lap_time_list[i][j], '-', color=color_pallet[colour_i], linewidth=1, label=labels[i])
                
        plt.xlabel("Training Steps (x1000)")
        # plt.title(f"{map_name.upper()}", size=10)
        plt.xlim(0, n_train_steps)
        plt.grid(True)

    plt.ylabel("Lap times (s)")
    h, l = plt.gca().get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3)
    # fig.legend(h[::3], l[::3], loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3)
    
    name = f"Data/Imgs/TrainingLapTimesComparison_{set_number}"
    std_img_saving(name)





compare_training()


