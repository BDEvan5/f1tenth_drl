
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from F1TenthRacingDRL.Utils.utils import *
from F1TenthRacingDRL.DataTools.plotting_utils import *


def make_training_crashes_plot():
    set_number = 2
    base_path = f"Data/FinalExperiment_{set_number}/"
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end"]
    
    max_speed = 8
    general_id = "TAL"
    n_repeats = 5
    n_train_steps = 60
    algorithm = "TD3"

    fig, axs = plt.subplots(1, 3, figsize=(5, 2.1), sharey=True)
    iteration = 0
    # map_name = "gbr"
    map_name = "mco"
    steps_list = []
    progresses_list = []
    crash_list = [] # store the train steps at which crashes took place
    for a, vehicle_key in enumerate(vehicle_keys):
        steps_list.append([])
        progresses_list.append([])
        crash_list.append([])

        for j in range(n_repeats):
            path = base_path + f"AgentOff_{algorithm}_{vehicle_key}_{map_name}_{general_id}_{max_speed}_{set_number}_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths) / 1000
            avg_progress = true_moving_average(progresses[:-1], 30)
            steps_list[a].append(steps)
            progresses_list[a].append(avg_progress)
            
            
        plt.sca(axs[a])
        mins = np.min(crash_list[a], axis=0)
        means = np.mean(crash_list[a], axis=0)
        maxes = np.max(crash_list[a], axis=0)
        plt.bar(xs, means, color=color_pallet[a], width=4, alpha=0.6)
        plot_error_bars_single_colour(xs, mins, maxes, color_pallet[a], 0.4, False)
        
        plt.xlim(-1, 61)
        plt.title(f"{labels[a]}", size=10)
        plt.grid(True)
        plt.yscale('log')
        axs[a].get_xaxis().set_major_locator(MultipleLocator(20))

    axs[1].set_xlabel("Training Steps (x1000)")
    axs[0].set_ylabel("# crashes")
    
    name = f"{base_path}Imgs/TrainingCrashComparison_{set_number}"
    std_img_saving(name)



make_training_crashes_plot()


