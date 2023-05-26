
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from RacingDRL.Utils.utils import *
from RacingDRL.DataTools.plotting_utils import *


def make_training_reward_plot():
    set_number = 1
    base_path = f"Data/FinalExperiment_{set_number}/"
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
        rewards_list = []
        for a, vehicle_key in enumerate(vehicle_keys):
            steps_list.append([])
            rewards_list.append([])
            for j in range(n_repeats):
                path = base_path + f"AgentOff_SAC_{vehicle_key}_{map_name}_{general_id}_{max_speed}_{set_number}_{j}/"
                rewards, lengths, progresses, _ = load_csv_data(path)
                steps = np.cumsum(lengths[:-1]) / 1000
                avg_reward = true_moving_average(rewards[:-1], 30)
                steps_list[a].append(steps)
                rewards_list[a].append(avg_reward)

        plt.sca(axs[m])
        xs = np.linspace(0, n_train_steps, 300)
        for i in range(len(steps_list)):
            colour_i = i + 0
            min, max, mean = convert_to_min_max_avg(steps_list[i], rewards_list[i], xs)
            plt.plot(xs, mean, '-', color=color_pallet[colour_i], linewidth=2, label=labels[i])
            plt.gca().fill_between(xs, min, max, color=color_pallet[colour_i], alpha=0.2)


        # axs[0].get_yaxis().set_major_locator(MultipleLocator(25))

        plt.xlabel("Training Steps (x1000)")
        plt.title(f"{map_name.upper()}", size=10)
        # plt.ylim(0, 100)
        plt.grid(True)

    axs[0].set_ylabel("Ep. Reward %")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3)
    
    name = f"{base_path}Imgs/TrainingRewardComparison_{set_number}"
    std_img_saving(name)





make_training_reward_plot()


