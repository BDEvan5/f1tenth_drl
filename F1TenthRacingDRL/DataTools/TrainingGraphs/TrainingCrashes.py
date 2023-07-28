
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from F1TenthRacingDRL.Utils.utils import *
from F1TenthRacingDRL.DataTools.plotting_utils import *


def make_training_crashes_plot():
    set_number = 4
    base_path = f"Data/FinalExperiment_{set_number}/"
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end"]
    
    max_speed = 8
    general_id = "TAL"
    n_repeats = 3
    n_train_steps = 60
    # algorithm = "TD3"
    algorithm = "SAC"

    fig, axs = plt.subplots(1, 3, figsize=(5, 2.1), sharey=True)
    iteration = 0
    map_name = "gbr"
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
            
            crashes = []
            for z in range(len(progresses)):
                if progresses[z] < 99:
                    crashes.append(steps[z])
            
            bin_sz = 4
            n_bin = int(60 / bin_sz)
            crash_bins = []
            xs = np.arange(0, 60, bin_sz) + bin_sz / 2
            crashes = np.array(crashes)
            for b in range(n_bin):
                crashes_to_come = np.sum(crashes > b * bin_sz)
                crashes_future = np.sum(crashes > (b+1) * bin_sz)
                n_crash = crashes_to_come - crashes_future
                crash_bins.append(n_crash)
                
            crash_list[a].append(crash_bins)
            
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
    
    name = f"{base_path}Imgs/TrainingCrashComparison_{map_name.upper()}_{set_number}"
    std_img_saving(name)


def make_training_crashes_plot_dual():
    set_number = 4
    base_path = f"Data/FinalExperiment_{set_number}/"
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end"]
    
    max_speed = 8
    general_id = "TAL"
    n_repeats = 3
    n_train_steps = 60
    algorithm_list = ["TD3", "SAC"]
    # algorithm = "TD3"
    # algorithm = "SAC"

    fig, axs = plt.subplots(2, 3, figsize=(5, 2.8), sharey=True)
    # map_name = "gbr"
    map_name = "mco"
    for alg, algorithm in enumerate(algorithm_list):
        steps_list = []
        progresses_list = []
        crash_list = [] # store the train steps at which crashes took place
        for a, vehicle_key in enumerate(vehicle_keys):
            steps_list.append([])
            progresses_list.append([])
            crash_list.append([])

            for j in range(n_repeats):
            # for j in range(2, 3):
                path = base_path + f"AgentOff_{algorithm}_{vehicle_key}_{map_name}_{general_id}_{max_speed}_{set_number}_{j}/"
                rewards, lengths, progresses, _ = load_csv_data(path)
                steps = np.cumsum(lengths) / 1000
                avg_progress = true_moving_average(progresses[:-1], 30)
                steps_list[a].append(steps)
                progresses_list[a].append(avg_progress)
                
                crashes = []
                for z in range(len(progresses)):
                    if progresses[z] < 99:
                        crashes.append(steps[z])
                
                bin_sz = 4
                n_bin = int(60 / bin_sz)
                crash_bins = []
                xs = np.arange(0, 60, bin_sz) + bin_sz / 2
                crashes = np.array(crashes)
                for b in range(n_bin):
                    crashes_to_come = np.sum(crashes > b * bin_sz)
                    crashes_future = np.sum(crashes > (b+1) * bin_sz)
                    n_crash = crashes_to_come - crashes_future
                    crash_bins.append(n_crash)
                    
                crash_list[a].append(crash_bins)
                
            plt.sca(axs[alg, a])
            mins = np.min(crash_list[a], axis=0)
            means = np.mean(crash_list[a], axis=0)
            maxes = np.max(crash_list[a], axis=0)
            plt.bar(xs, means, color=color_pallet[a], width=4, alpha=0.6)
            # plot_error_bars_single_colour(xs, mins, maxes, color_pallet[a], 0.4, False)
            
            plt.xlim(-1, 61)
            if alg == 0:
                plt.title(f"{labels[a]}", size=10)
            plt.grid(True)
            plt.yscale('log')
            axs[alg, a].get_xaxis().set_major_locator(MultipleLocator(20))

            axs[alg, a].text(48, 86, algorithm, fontdict={'fontsize': 10, 'fontweight':'bold'}, ha='center', bbox=dict(facecolor='white', alpha=0.99, boxstyle='round,pad=0.15', edgecolor='gainsboro'))

    axs[1, 1].set_xlabel("Training Steps (x1000)")
    axs[0, 0].set_ylabel("# crashes")
    axs[1, 0].set_ylabel("# crashes")
    
    name = f"{base_path}Imgs/TrainingCrashes_Algs_{map_name.upper()}_{set_number}"
    std_img_saving(name)



# make_training_crashes_plot()
make_training_crashes_plot_dual()

