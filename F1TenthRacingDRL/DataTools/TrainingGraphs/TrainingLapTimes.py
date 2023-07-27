
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from F1TenthRacingDRL.Utils.utils import *
from F1TenthRacingDRL.DataTools.plotting_utils import *



def make_training_laptimes_plot():
    set_number = 1
    base_path = f"Data/FinalExperiment_{set_number}/"
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end"]
    
    # map_list = ["gbr"]
    map_name = "gbr"
    # map_name = "mco"
    # algorithm = "SAC"
    # algorithm = "TD3"
    algorithm_list = ["SAC", "TD3"]
    
    max_speed = 8
    general_id = "TAL"
    n_repeats = 3
    n_train_steps = 60

    # fig, axs = plt.subplots(1, 1, figsize=(5, 1.7))
    # fig, axs = plt.subplots(1, 1, figsize=(5, 2.1))
    fig, axs = plt.subplots(2, 3, figsize=(5.3, 3.), sharey=True, sharex=True)
    
    for a_n, algorithm in enumerate(algorithm_list):
        steps_list = []
        lap_time_list = []
        for a, vehicle_key in enumerate(vehicle_keys):
            steps_list.append([])
            lap_time_list.append([])
            for j in range(n_repeats):
                path = base_path + f"AgentOff_{algorithm}_{vehicle_key}_{map_name}_{general_id}_{max_speed}_{set_number}_{j}/"
                rewards, lengths, progresses, _ = load_csv_data(path)
                steps = np.cumsum(lengths) / 1000
                
                lap_times, completed_steps = [], []
                for z in range(len(progresses)):
                    if progresses[z] > 99:
                        lap_times.append(lengths[z]/10)
                        completed_steps.append(steps[z])
                        
                steps_list[a].append(completed_steps)
                lap_time_list[a].append(lap_times)

        for itteration in range(n_repeats):
            for i in range(len(steps_list)):
                colour_i = i
                axs[a_n, i].plot(steps_list[i][itteration], lap_time_list[i][itteration], '.', color=color_pallet[colour_i], markersize=5, label=labels[i], alpha=0.6, linewidth=2)



    for z in range(3):
        axs[0, z].grid(True, axis='both')
        axs[1, z].grid(True, axis='both')
        axs[0, z].set_title(labels[z], size=11)

    # axs[0, 1].set_title("SAC", size=10, fontdict={'fontweight':'bold'})
    # axs[1, 1].set_title("TD3", size=10, fontdict={'fontweight':'bold'})

    # axs[0, 1].text(30, 85, "SAC", fontdict={'fontsize': 12, 'fontweight':'bold'}, ha='center')
    # axs[1, 1].text(30, 85, "TD3", fontdict={'fontsize': 12, 'fontweight':'bold'}, ha='center')
    axs[0, 0].text(45, 82, "SAC", fontdict={'fontsize': 13, 'fontweight':'bold'}, ha='center', bbox=dict(facecolor='white', alpha=0.99, boxstyle='round,pad=0.25', edgecolor='grey'))
    axs[1, 0].text(45, 82, "TD3", fontdict={'fontsize': 13, 'fontweight':'bold'}, ha='center', bbox=dict(facecolor='white', alpha=0.99, boxstyle='round,pad=0.25', edgecolor='grey'))

    axs[1, 1].set_xlabel("Training Steps (x1000)")
    # plt.title(f"{map_name.upper()}", size=10)
    plt.xlim(0, n_train_steps)
    plt.grid(True)
    # plt.gca().yaxis.set_major_locator(MultipleLocator(30))

    axs[1, 0].set_ylabel("Lap times (s)")
    axs[0, 0].set_ylabel("Lap times (s)")
    # h, l = plt.gca().get_legend_handles_labels()
    # fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3)
    # fig.legend(h[:3], l[:3], loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3)
    
    name = f"{base_path}Imgs/TrainingLapTimesArchitectures_{set_number}"
    std_img_saving(name)





make_training_laptimes_plot()


