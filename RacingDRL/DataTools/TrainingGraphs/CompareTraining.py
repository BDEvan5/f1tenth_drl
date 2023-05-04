
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from RacingDRL.Utils.utils import *
from RacingDRL.DataTools.plotting_utils import *


def compare_training():
    base_path = "Data/"
    set_number = 8
    folder_keys = ["PlanningMaps", "TrajectoryMaps", "EndMaps"]
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end"]
    
    map_list = ["mco", "gbr"]
    
    max_speed = 8
    general_id = "train"
    n_repeats = 3

    fig, axs = plt.subplots(1, 2, figsize=(5, 2.1))
    
    for m, map_name in enumerate(map_list):
        steps_list = []
        progresses_list = []
        for a, architecture in enumerate(folder_keys):
            p = base_path + architecture + f"_{set_number}/"
            steps_list.append([])
            progresses_list.append([])
            for j in range(n_repeats):
                path = p + f"AgentOff_SAC_{vehicle_keys[a]}_{map_name}_{general_id}_{max_speed}_{set_number}_{j}/"
                rewards, lengths, progresses, _ = load_csv_data(path)
                steps = np.cumsum(lengths[:-1]) / 1000
                avg_progress = true_moving_average(progresses[:-1], 20)
                steps_list[a].append(steps)
                progresses_list[a].append(avg_progress)

        plt.sca(axs[m])
        xs = np.linspace(0, 30, 300)
        for i in range(len(steps_list)):
            colour_i = i + 1
            min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], xs)
            plt.plot(xs, mean, '-', color=pp_dark[colour_i], linewidth=2, label=labels[i])
            plt.gca().fill_between(xs, min, max, color=pp_dark[colour_i], alpha=0.2)


        # axs[0].get_yaxis().set_major_locator(MultipleLocator(25))

        plt.xlabel("Training Steps (x1000)")
        plt.title(f"{map_name.upper()}", size=10)
        plt.ylim(0, 100)
        plt.grid(True)

    axs[0].set_ylabel("Track Progress %")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3)
    
    name = f"Data/TrainingComparision_{set_number}"
    std_img_saving(name)


def make_TrainingGraph():
    base_path = "Data/"
    # test_name = "TrajectoryMaps" 
    # architecture = "TrajectoryFollower"
    # test_name = "PlanningMaps" 
    # architecture = "Game"
    test_name = "EndMaps"
    architecture = "endToEnd"
    set_number = 8
    p = base_path + test_name + f"_{set_number}/"
    max_speed = 8
    general_id = "train"
    # general_id = "v6"

    steps_list = []
    progresses_list = []
    
    map_names = ['gbr', "mco"]
    # map_names = ["aut", "esp"]
    # map_names = ["aut", "esp", "gbr", "mco"]

    n_repeats = 3
    for i, id_name in enumerate(map_names): 
        steps_list.append([])
        progresses_list.append([])
        for j in range(n_repeats):
            path = p + f"AgentOff_SAC_{architecture}_{id_name}_{general_id}_{max_speed}_{set_number}_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 1000
            avg_progress = true_moving_average(progresses[:-1], 20)
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    plt.figure(2, figsize=(4.5, 2.3))

    xs = np.linspace(0, 30, 300)
    for i in range(len(steps_list)):
        min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], xs)
        plt.plot(xs, mean, '-', color=pp_dark[i], linewidth=2, label=map_names[i])
        plt.gca().fill_between(xs, min, max, color=pp_dark[i], alpha=0.2)


    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x1000)")
    plt.ylabel("Track Progress %")
    plt.ylim(0, 100)
    # plt.legend(loc='center', bbox_to_anchor=(1.06, 0.5), ncol=1)
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.52), ncol=4)
    plt.tight_layout()
    plt.grid()

    name = p + f"{test_name}_TrainingGraph"
    std_img_saving(name)


compare_training()


