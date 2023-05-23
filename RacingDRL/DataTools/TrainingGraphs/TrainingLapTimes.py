
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from RacingDRL.Utils.utils import *
from RacingDRL.DataTools.plotting_utils import *



def compare_training():
    set_number = 1
    base_path = f"Data/FinalExperiment_{set_number}/"
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end"]
    
    # map_list = ["gbr"]
    map_name = "mco"
    
    max_speed = 8
    general_id = "TAL"
    n_repeats = 5
    n_train_steps = 60

    fig, axs = plt.subplots(1, 1, figsize=(5, 2.1))
    
    steps_list = []
    lap_time_list = []
    for a, vehicle_key in enumerate(vehicle_keys):
        steps_list.append([])
        lap_time_list.append([])
        for j in range(n_repeats):
            path = base_path + f"AgentOff_SAC_{vehicle_key}_{map_name}_{general_id}_{max_speed}_{set_number}_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths) / 1000
            
            lap_times, completed_steps = [], []
            for z in range(len(progresses)):
                if progresses[z] > 99:
                    lap_times.append(lengths[z]/10)
                    completed_steps.append(steps[z])
                    
            steps_list[a].append(completed_steps)
            lap_time_list[a].append(lap_times)

    itteration = 0
    for i in range(len(steps_list)):
        colour_i = i
        
        plt.plot(steps_list[i][itteration], lap_time_list[i][itteration], '.-', color=color_pallet[colour_i], markersize=3, label=labels[i], alpha=0.6, linewidth=2)
                
    plt.xlabel("Training Steps (x1000)")
    # plt.title(f"{map_name.upper()}", size=10)
    plt.xlim(0, n_train_steps)
    plt.grid(True)

    plt.ylabel("Lap times (s)")
    h, l = plt.gca().get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3)
    # fig.legend(h[::3], l[::3], loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3)
    
    name = f"{base_path}Imgs/TrainingLapTimesComparison_{set_number}"
    std_img_saving(name)





compare_training()


