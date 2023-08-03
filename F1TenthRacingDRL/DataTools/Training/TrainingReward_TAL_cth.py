
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from F1TenthRacingDRL.Utils.utils import *
from F1TenthRacingDRL.DataTools.plotting_utils import *


def make_training_reward_plot():
    set_number = 1
    base_path = f"Data/Experiment_{set_number}/"
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end"]
    
    map_name = "gbr"
    reward_signals = ["Cth", "TAL"]
    
    max_speed = 8
    # general_id = "TAL"
    n_repeats = 3
    algorithm = "SAC"
    # algorithm = "TD3"
    n_train_steps = 60

    fig, axs = plt.subplots(2, 2, figsize=(5, 3), sharex=True)
    
    for m, general_id in enumerate(reward_signals):
        steps_list = []
        progress_list = []
        rewards_list = []
        for a, vehicle_key in enumerate(vehicle_keys):
            steps_list.append([])
            progress_list.append([])
            rewards_list.append([])
            for j in range(n_repeats):
                path = base_path + f"AgentOff_{algorithm}_{vehicle_key}_{map_name}_{general_id}_{max_speed}_{set_number}_{j}/"
                rewards, lengths, progresses, _ = load_csv_data(path)
                steps = np.cumsum(lengths[:-1]) / 1000
                new_steps, avg_progress = true_moving_average_steps(steps, progresses[:-1], 50, 1000)
                new_steps, avg_reward = true_moving_average_steps(steps, rewards[:-1], 50, 1000)
                steps_list[a].append(new_steps)
                progress_list[a].append(avg_progress)
                rewards_list[a].append(avg_reward)


        plt.sca(axs[0, m])
        xs = np.linspace(0, n_train_steps, 300)
        for i in range(len(steps_list)):
            colour_i = i + 0
            min, max, mean = convert_to_min_max_avg(steps_list[i], progress_list[i], xs)
            plt.plot(xs, mean, '-', color=color_pallet[colour_i], linewidth=2, label=labels[i])
            plt.gca().fill_between(xs, min, max, color=color_pallet[colour_i], alpha=0.2)
        plt.grid(True)
        plt.title(f"{general_id.upper()}", size=10)

        plt.sca(axs[1, m])
        xs = np.linspace(0, n_train_steps, 300)
        for i in range(len(steps_list)):
            colour_i = i + 0
            min, max, mean = convert_to_min_max_avg(steps_list[i], rewards_list[i], xs)
            plt.plot(xs, mean, '-', color=color_pallet[colour_i], linewidth=2, label=labels[i])
            plt.gca().fill_between(xs, min, max, color=color_pallet[colour_i], alpha=0.2)
        plt.grid(True)

        plt.xlabel("Training Steps (x1000)")

    axs[0, 0].yaxis.set_major_locator(plt.MaxNLocator(5))
    axs[1, 0].yaxis.set_major_locator(plt.MaxNLocator(5))
    axs[0, 1].yaxis.set_major_locator(plt.MaxNLocator(5))
    axs[1, 1].yaxis.set_major_locator(plt.MaxNLocator(5))

    axs[0, 0].set_ylabel("Average \nprogress (%)")
    axs[1, 0].set_ylabel("Episode \nreward %")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 0.94), ncol=3)
    
    name = f"{base_path}Imgs/TrainingRewardProgress_Rsig_{set_number}"
    std_img_saving(name)





make_training_reward_plot()


