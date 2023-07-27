
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from F1TenthRacingDRL.Utils.utils import *
from F1TenthRacingDRL.DataTools.plotting_utils import *


def make_training_progress_plot():
    set_number = 1
    base_path = f"Data/FinalExperiment_{set_number}/"
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end"]
    
    map_name = "gbr"
    alg_list = ["SAC", "TD3", "DDPG"]
    
    max_speed = 8
    general_id = "TAL"
    n_repeats = 3
    n_train_steps = 60

    fig, axs = plt.subplots(2, 3, figsize=(7, 3.5))
    
    for m, alg in enumerate(alg_list):
        steps_list = []
        progresses_list = []
        rewards_list = []
        for a, vehicle_key in enumerate(vehicle_keys):
            steps_list.append([])
            progresses_list.append([])
            rewards_list.append([])
            for j in range(n_repeats):
                path = base_path + f"AgentOff_{alg}_{vehicle_key}_{map_name}_{general_id}_{max_speed}_{set_number}_{j}/"
                rewards, lengths, progresses, _ = load_csv_data(path)
                steps = np.cumsum(lengths[:-1]) / 1000
                new_steps, avg_progress = true_moving_average_steps(steps, progresses[:-1], 80, 1000)
                new_steps, avg_rewards = true_moving_average_steps(steps, rewards[:-1], 80, 1000)
                steps_list[a].append(new_steps)
                progresses_list[a].append(avg_progress)
                rewards_list[a].append(avg_rewards)

        plt.sca(axs[0, m])
        xs = np.linspace(0, n_train_steps, 300)
        for i in range(len(steps_list)):
            colour_i = i + 0
            min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], xs)
            plt.plot(xs, mean, '-', color=color_pallet[colour_i], linewidth=2, label=labels[i])
            plt.gca().fill_between(xs, min, max, color=color_pallet[colour_i], alpha=0.2)

        plt.ylim(0, 100)
        plt.grid(True)

        plt.sca(axs[1, m])
        xs = np.linspace(0, n_train_steps, 300)
        for i in range(len(steps_list)):
            colour_i = i + 0
            min, max, mean = convert_to_min_max_avg(steps_list[i], rewards_list[i], xs)
            plt.plot(xs, mean, '-', color=color_pallet[colour_i], linewidth=2, label=labels[i])
            plt.gca().fill_between(xs, min, max, color=color_pallet[colour_i], alpha=0.2)


        plt.xlabel("Training Steps (x1000)")
        plt.title(f"{alg}", size=10)
        plt.grid(True)

    axs[0, 0].set_ylabel("Track Progress %")
    axs[1, 0].set_ylabel("Episode Reward")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    
    name = base_path +  f"Imgs/Training_Algorithms_{set_number}"
    std_img_saving(name)





make_training_progress_plot()


