import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv

from RacingDRL.DataTools.MapData import MapData
from RacingDRL.DataTools.plotting_utils import *

root_path = "/home/benjy/sim_ws/src/f1tenth_racing/Data/"


def make_path_overlays():
    agent_names = ["AgentOff_SAC_Game_mco_TAL_8_1_0", "AgentOff_SAC_TrajectoryFollower_mco_TAL_8_1_0", "AgentOff_SAC_endToEnd_mco_TAL_8_1_0", "PurePursuit"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]
    # labels = ["Full\nplanning", "Trajectory\ntracking", "End-to-end", "Classic"]
    real_runs = [0, 1, 1, 0]
    sim_runs = [0, 0, 0, 0]
    real_folder = "ResultsJetson24/"
    sim_folder = "ResultsRos24/"

    real_states = []
    sim_states = []

    map_data = MapData("CornerHall") 

    for i in range(4):
        folder = root_path + real_folder + f"{agent_names[i]}/Run_{real_runs[i]}"
        with open(folder + f"/Run_{real_runs[i]}_states.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)
        real_states.append(states)
        

        folder = root_path + sim_folder + f"{agent_names[i]}/Run_{sim_runs[i]}"
        with open(folder + f"/Run_{sim_runs[i]}_states.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)
        sim_states.append(states)


    plt.figure(1)
    plt.clf()
    map_data.plot_map_img_transpose()

    for i in range(4):
        xs, ys = map_data.xy2rc(real_states[i][:, 0], real_states[i][:, 1])
        plt.plot(ys, xs, label=labels[i], color=color_pallet[i], linewidth=2)

    plt.xlim(20, 380)
    plt.ylim(10, 135)
    plt.legend(ncol=2, loc='lower left', fontsize=12)
    plt.xticks([])
    plt.yticks([])

    name = "Sim2Real/Imgs/RealWorldPathOverlays"
    std_img_saving(name)

make_path_overlays()