
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from F1TenthRacingDRL.Utils.utils import *
from F1TenthRacingDRL.DataTools.plotting_utils import *


def make_TrainingGraph():
    base_path = "Data/"
    test_name = "FinalExperiment" 
    # test_name = "PreTrained" 
    set_number = 4
    p = base_path + test_name + f"_{set_number}/"
    max_speed = 8
    general_id = "TAL"
    # map_name = "gbr"
    map_name = "mco"
    algorithm = "TD3"
    # algorithm = "SAC"

    steps_list = []
    progresses_list = []
    
    arch_names = ["endToEnd", "TrajectoryFollower", "Game"]
    # arch_names = ["Game"]
    # arch_names = ["Game", "TrajectoryFollower"]
    labels = ["End-to-End", "Trajectory tracking", "Full planning"]

    n_repeats = 3
    for i, architecture in enumerate(arch_names): 
        steps_list.append([])
        progresses_list.append([])
        # for j in range(2, 3):
        for j in range(n_repeats):
            path = p + f"AgentOff_{algorithm}_{architecture}_{map_name}_{general_id}_{max_speed}_{set_number}_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 1000
            avg_progress = true_moving_average(progresses[:-1], 30)
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    plt.figure(2, figsize=(4.5, 2.3))

    # xs = np.linspace(0, 80, 300)
    xs = np.linspace(0, 60, 300)
    # xs = np.linspace(0, 30, 300)
    for i in range(len(steps_list)):
        min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], xs)
        plt.plot(xs, mean, '-', color=color_pallet[i], linewidth=2, label=labels[i])
        plt.gca().fill_between(xs, min, max, color=color_pallet[i], alpha=0.2)


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
    print(f"Finneshed")


def make_TrainingGraphReward():
    base_path = "Data/"
    # test_name = "TrajectoryMaps" 
    # architecture = "TrajectoryFollower"
    test_name = "PlanningMaps" 
    architecture = "Game"
    # test_name = "EndMaps"
    # architecture = "endToEnd"
    set_number = 5
    p = base_path + test_name + f"_{set_number}/"
    max_speed = 8
    general_id = "TAL"
    # general_id = "train"
    # general_id = "v6"

    steps_list = []
    progresses_list = []
    
    # map_names = ['gbr']
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
            avg_progress = true_moving_average(rewards[:-1], 20)
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    plt.figure(2, figsize=(4.5, 2.3))
    plt.clf()

    # xs = np.linspace(0, 80, 300)
    xs = np.linspace(0, 60, 300)
    # xs = np.linspace(0, 30, 300)
    for i in range(len(steps_list)):
        min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], xs)
        plt.plot(xs, mean, '-', color=pp_dark[i], linewidth=2, label=map_names[i])
        plt.gca().fill_between(xs, min, max, color=pp_dark[i], alpha=0.2)


    # plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x1000)")
    plt.ylabel("Ep. Reward")
    # plt.ylim(0, 100)
    # plt.legend(loc='center', bbox_to_anchor=(1.06, 0.5), ncol=1)
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.52), ncol=4)
    plt.tight_layout()
    plt.grid()

    name = p + f"{test_name}_TrainingReward"
    std_img_saving(name)


make_TrainingGraph()
# make_TrainingGraphReward()


