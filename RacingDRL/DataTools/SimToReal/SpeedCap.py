import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv

from RacingDRL.DataTools.MapData import MapData
from RacingDRL.DataTools.plotting_utils import *

root_path = "/home/benjy/sim_ws/src/f1tenth_racing/Data/"

def make_laptime_barplot23():
    agent_names = ["AgentOff_SAC_Game_mco_TAL_8_1_0", "AgentOff_SAC_TrajectoryFollower_mco_TAL_8_1_0", "AgentOff_SAC_endToEnd_mco_TAL_8_1_0", "PurePursuit"]
    labels = ["Full\n planning", "Trajectory\n tracking", "End-to-end", "Classic"]
    real_runs_2 = [0, 1, 1, 0]
    sim_runs_2 = [0, 0, 0, 0]
    real_runs_3 = [1, 0, 0, 1]
    sim_runs_3 = [1, 1, 1, 1]
    real_folder = "ResultsJetson24/"
    sim_folder = "ResultsRos24/"

    real_lap_times_2 = []
    sim_lap_times_2 = []
    real_lap_times_3 = []
    sim_lap_times_3 = []
    for i in range(4):
        folder = root_path + real_folder + f"{agent_names[i]}/Run_{real_runs_2[i]}"
        with open(folder + f"/Run_{real_runs_2[i]}_states.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)

        lap_time = len(states) * 0.1
        real_lap_times_2.append(lap_time)

        folder = root_path + sim_folder + f"{agent_names[i]}/Run_{sim_runs_2[i]}"
        with open(folder + f"/Run_{sim_runs_2[i]}_states.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)

        lap_time = len(states) * 0.1
        sim_lap_times_2.append(lap_time)

        folder = root_path + real_folder + f"{agent_names[i]}/Run_{real_runs_3[i]}"
        with open(folder + f"/Run_{real_runs_3[i]}_states.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)

        lap_time = len(states) * 0.1
        real_lap_times_3.append(lap_time)

        folder = root_path + sim_folder + f"{agent_names[i]}/Run_{sim_runs_3[i]}"
        with open(folder + f"/Run_{sim_runs_3[i]}_states.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)

        lap_time = len(states) * 0.1
        sim_lap_times_3.append(lap_time)

    xs = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.3))
    axes[0].bar(xs - width/2, sim_lap_times_2, color=color_pallet, width=width, alpha=0.5, label="Sim")
    axes[0].bar(xs + width/2, real_lap_times_2, color=color_pallet, width=width, label="Real", hatch='//')

    axes[1].bar(xs - width/2, sim_lap_times_3, color=color_pallet, width=width, alpha=0.5, label="Sim")
    axes[1].bar(xs + width/2, real_lap_times_3, color=color_pallet, width=width, label="Real", hatch='//')

    axes[0].set_title("2 m/s")
    axes[0].set_ylabel("Lap time [s]")
    axes[0].set_xticks(xs, labels, fontsize=7)
    axes[0].grid(True)

    axes[1].set_title("3 m/s")
    axes[1].set_ylabel("Lap time [s]")
    axes[1].set_xticks(xs, labels, fontsize=7)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='center', bbox_to_anchor=(0.5, 0.9) ,ncol=2, fontsize=9)

    name = "Sim2Real/Imgs/LapTimes2p3"
    std_img_saving(name)


def make_laptime_endToEnd():
    agent_name = "AgentOff_SAC_endToEnd_mco_TAL_8_1_0"
    labels = ["End-to-end"]
    real_runs = [0, 1, 2]
    real_folder = "ResultsJetson24_2/"

    real_lap_times = []
    folder = root_path + "ResultsJetson24/" + f"{agent_name}/Run_1"
    with open(folder + f"/Run_1_states.csv") as file:
        state_reader = csv.reader(file, delimiter=',')
        state_list = []
        for row in state_reader:
            state_list.append(row)
        states = np.array(state_list[1:]).astype(float)

        lap_time = len(states) * 0.1
        real_lap_times.append(lap_time)

    for i in range(3):
        folder = root_path + real_folder + f"{agent_name}/Run_{real_runs[i]}"
        with open(folder + f"/Run_{real_runs[i]}_states.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)

        lap_time = len(states) * 0.1
        real_lap_times.append(lap_time)

    xs = np.arange(len(real_runs)+1)
    width = 0.35

    fig, axes = plt.subplots(1, 1, figsize=(6, 2.3))
    # axes[0].bar(xs - width/2, sim_lap_times_2, color=color_pallet, width=width, alpha=0.5, label="Sim")
    axes.bar(xs + width/2, real_lap_times, color=color_pallet[2], width=width, label="Real", hatch='//')


    # axes[0].set_title("2 m/s")
    # axes[0].set_ylabel("Lap time [s]")
    # axes[0].set_xticks(xs, labels, fontsize=7)
    # axes[0].grid(True)

    # axes[1].set_title("3 m/s")
    # axes[1].set_ylabel("Lap time [s]")
    # axes[1].set_xticks(xs, labels, fontsize=7)

    # h, l = axes[0].get_legend_handles_labels()
    # fig.legend(h, l, loc='center', bbox_to_anchor=(0.5, 0.9) ,ncol=2, fontsize=9)

    name = "Sim2Real/Imgs/LapTimesEndToEnd"
    std_img_saving(name)



# make_laptime_barplot23()
make_laptime_endToEnd()

