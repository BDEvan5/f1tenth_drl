
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

    fig, axs = plt.subplots(1, 3, figsize=(7, 2.1))
    
    for m, alg in enumerate(alg_list):
        steps_list = []
        progresses_list = []
        for a, vehicle_key in enumerate(vehicle_keys):
            steps_list.append([])
            progresses_list.append([])
            for j in range(n_repeats):
                path = base_path + f"AgentOff_{alg}_{vehicle_key}_{map_name}_{general_id}_{max_speed}_{set_number}_{j}/"
                rewards, lengths, progresses, _ = load_csv_data(path)
                steps = np.cumsum(lengths[:-1]) / 1000
                avg_progress = true_moving_average(progresses[:-1], 30)
                steps_list[a].append(steps)
                progresses_list[a].append(avg_progress)

        plt.sca(axs[m])
        xs = np.linspace(0, n_train_steps, 300)
        for i in range(len(steps_list)):
            colour_i = i + 0
            min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], xs)
            plt.plot(xs, mean, '-', color=color_pallet[colour_i], linewidth=2, label=labels[i])
            plt.gca().fill_between(xs, min, max, color=color_pallet[colour_i], alpha=0.2)


        # axs[0].get_yaxis().set_major_locator(MultipleLocator(25))

        plt.xlabel("Training Steps (x1000)")
        plt.title(f"{alg}", size=10)
        plt.ylim(0, 100)
        plt.grid(True)

    axs[0].set_ylabel("Track Progress %")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3)
    
    name = base_path +  f"Imgs/TrainingProgressComparison_Algs_{set_number}"
    std_img_saving(name)





make_training_progress_plot()


