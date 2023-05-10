
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from RacingDRL.Utils.utils import *
from RacingDRL.DataTools.plotting_utils import *


def make_TrainingGraph():
    base_path = "Data/"
    # test_name = "Comparison" 
    # set_number = 5
    test_name = "ComparativeAnalysis" 
    set_number = "GBR"
    p = base_path + test_name + f"_{set_number}/"
    set_number = 5

    steps_list = []
    progresses_list = []
    id_key = "TAL"
    
    id_names = ["Game", "TrajectoryFollower", "endToEnd"]
    # id_names = ["Game", "TrajectoryFollower"]

    n_repeats = 1
    for i, id_name in enumerate(id_names): 
        steps_list.append([])
        progresses_list.append([])
        for j in range(n_repeats):
            path = p + f"AgentOff_SAC_{id_name}_gbr_{id_key}_8_{set_number}_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 1000
            avg_progress = true_moving_average(progresses[:-1], 20)
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    plt.figure(2, figsize=(4.5, 2.3))

    xs = np.linspace(0, 60, 300)
    for i in range(len(steps_list)):
        min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], xs)
        plt.plot(xs, mean, '-', color=pp_dark[i], linewidth=2, label=id_names[i])
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
    
    steps_list = []
    progresses_list = []
    for i, id_name in enumerate(id_names): 
        steps_list.append([])
        progresses_list.append([])
        for j in range(n_repeats):
            path = p + f"AgentOff_SAC_{id_name}_gbr_{id_key}_8_{set_number}_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 1000
            avg_progress = true_moving_average(rewards[:-1], 30)
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    plt.figure(2, figsize=(4.5, 2.3))
    plt.clf()

    xs = np.linspace(0, 60, 300)
    for i in range(len(steps_list)):
        min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], xs)
        plt.plot(xs, mean, '-', color=pp_dark[i], linewidth=2, label=id_names[i])
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

    steps_list = []
    progresses_list = []
    for i, id_name in enumerate(id_names): 
        steps_list.append([])
        progresses_list.append([])
        for j in range(n_repeats):
            path = p + f"AgentOff_SAC_{id_name}_gbr_{id_key}_8_{set_number}_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths[:-1]) / 1000
            avg_progress = true_moving_average(lengths[:-1], 30)
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    plt.figure(2, figsize=(4.5, 2.3))
    plt.clf()

    xs = np.linspace(0, 60, 300)
    for i in range(len(steps_list)):
        min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], xs)
        plt.plot(xs, mean, '-', color=pp_dark[i], linewidth=2, label=id_names[i])
        plt.gca().fill_between(xs, min, max, color=pp_dark[i], alpha=0.2)


    # plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x1000)")
    plt.ylabel("Ep. Lengths")
    # plt.ylim(0, 100)
    # plt.legend(loc='center', bbox_to_anchor=(1.06, 0.5), ncol=1)
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.52), ncol=4)
    plt.tight_layout()
    plt.grid()

    name = p + f"{test_name}_TrainingLengths"
    std_img_saving(name)


make_TrainingGraph()


