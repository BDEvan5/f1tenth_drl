import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv

from RacingDRL.DataTools.MapData import MapData
from RacingDRL.DataTools.plotting_utils import *

root_path = "/home/benjy/sim_ws/src/f1tenth_racing/Data/"


def make_laptime_barplot():
    agent_names = ["AgentOff_SAC_Game_mco_TAL_8_1_0", "AgentOff_SAC_TrajectoryFollower_mco_TAL_8_1_0", "AgentOff_SAC_endToEnd_mco_TAL_8_1_0", "PurePursuit"]
    labels = ["Full\n planning", "Trajectory\n tracking", "End-to-end", "Classic"]
    real_runs = [0, 1, 1, 0]
    sim_runs = [0, 0, 0, 0]
    real_folder = "ResultsJetson24/"
    sim_folder = "ResultsRos24/"

    real_lap_times = []
    sim_lap_times = []
    for i in range(4):
        folder = root_path + real_folder + f"{agent_names[i]}/Run_{real_runs[i]}"
        with open(folder + f"/Run_{real_runs[i]}_states.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)

        lap_time = len(states) * 0.1
        real_lap_times.append(lap_time)

        folder = root_path + sim_folder + f"{agent_names[i]}/Run_{sim_runs[i]}"
        with open(folder + f"/Run_{sim_runs[i]}_states.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)

        lap_time = len(states) * 0.1
        sim_lap_times.append(lap_time)

    xs = np.arange(len(labels))
    width = 0.35

    plt.figure(1, figsize=(5, 2.3))
    plt.bar(xs - width/2, sim_lap_times, color=color_pallet, width=width, alpha=0.5, label="Sim")
    plt.bar(xs + width/2, real_lap_times, color=color_pallet, width=width, label="Real", hatch='//')

    plt.ylabel("Lap time [s]")
    # plt.xlabel("Agent")
    plt.xticks(xs, labels)
    plt.legend(ncol=2)

    name = "Sim2Real/Imgs/LapTimes"
    std_img_saving(name)

make_laptime_barplot()


