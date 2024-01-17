import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv

from f1tenth_drl.DataTools.MapData import MapData
from f1tenth_drl.DataTools.plotting_utils import *

root_path = "/home/benjy/sim_ws/src/f1tenth_racing/Data/"


def make_agent_trajectory(agent_name, label_name, folder, run_n):
    map_data = MapData("CornerHall") 

    folder = root_path + folder + f"{agent_name}/Run_{run_n}"
    with open(folder + f"/Run_{run_n}_states.csv") as file:
        state_reader = csv.reader(file, delimiter=',')
        state_list = []
        for row in state_reader:
            state_list.append(row)
    states = np.array(state_list[1:]).astype(float)
        
    plt.figure(1)
    plt.clf()
    map_data.plot_map_img_transpose()

    xs, ys = map_data.xy2rc(states[:, 0], states[:, 1])
    points = np.array([ys, xs]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(1, 5)
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(states[:, 3])
    lc.set_linewidth(3)
    line = plt.gca().add_collection(lc)
    plt.colorbar(line, fraction=0.046, pad=0.04, shrink=0.48, label="Speed (m/s)")

    plt.text(60, 20, label, fontsize=20, color='black')
    plt.xlim(35, 380)

    plt.xticks([])
    plt.yticks([])

    label_name = label_name.replace("-", "_")
    label_name = label_name.replace("/", "_")
    label_name = label_name.replace(" ", "_")
    name = f"Sim2Real/Imgs/Trajectory_{label_name}"
    std_img_saving(name)



agent_name = "AgentOff_SAC_endToEnd_mco_TAL_8_1_0"
label = "End-to-end - 5 m/s"
folder = "ResultsJetson24_2/"
run_n = 2
make_agent_trajectory(agent_name, label, folder, run_n)

# agent_name = "AgentOff_SAC_Game_mco_TAL_8_1_0"
# label = "Full planning - 4 m/s"
# folder = "ResultsJetson24_2/"
# run_n = 4
# make_agent_trajectory(agent_name, label, folder, run_n)


# agent_name = "AgentOff_SAC_TrajectoryFollower_mco_TAL_8_1_0"
# label = "Trajectory tracking - 2 m/s"
# folder = "ResultsJetson23/"
# run_n = 2
# make_agent_trajectory(agent_name, label, folder, run_n)




