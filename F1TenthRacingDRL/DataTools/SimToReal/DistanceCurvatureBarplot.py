import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv

from FTenthRacingDRL.DataTools.MapData import MapData
from FTenthRacingDRL.DataTools.plotting_utils import *

root = "/home/benjy/sim_ws/src/f1tenth_racing/Data/"

class PathResults:
    def __init__(self, agent_name, folder, run_list=[0, 1, 2, 3, 4]):
        self.agent_name = agent_name
        self.folder = folder
        self.run_list = run_list

        self.states = []
        for run_n in run_list:
            self.load_state_data(run_n)

    def load_state_data(self, run_n):
        folder = root + self.folder + f"{self.agent_name}/Run_{run_n}"
        with open(folder + f"/Run_{run_n}_states.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)
        self.states.append(states)

    def calculate_distance_list(self):
        distances = []
        for i in range(len(self.states)):
            pts = self.states[i][:, 0:2]
            dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            distances.append(np.sum(dists))

        return distances
    
    def calculate_curvatures_list(self):
        total_curvatures = []
        mean_curvatures = []
        for i in range(len(self.states)):
            pts = self.states[i][:, 0:2]
            dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            ths = self.states[i][:, 2]
            dts = ths[1:] - ths[:-1]
            dts = dts[dists > 0.05]
            dists = dists[dists > 0.05]
            curvatures = np.abs(dts / dists)
            total_curvatures.append(np.sum(curvatures))
            mean_curvatures.append(np.mean(curvatures))

        return total_curvatures, mean_curvatures



def make_distance_curvature_plot():
    agent_names = ["AgentOff_SAC_Game_mco_TAL_8_1_0", "AgentOff_SAC_TrajectoryFollower_mco_TAL_8_1_0", "AgentOff_SAC_endToEnd_mco_TAL_8_1_0", "PurePursuit"]
    labels = ["Full\n planning", "Trajectory\n tracking", "End-to-end", "Classic"]
    
    real_folder = "ResultsJetson25/"
    sim_folder = "ResultsRos25/"

    real_data = [PathResults(agent_names[i], real_folder) for i in range(4)]
    sim_data = [PathResults(agent_names[i], sim_folder) for i in range(4)]


    xs = np.arange(len(labels))
    width = 0.4

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 2.2))
    for i in range(len(real_data)):
        real_distances = real_data[i].calculate_distance_list()
        sim_distances = sim_data[i].calculate_distance_list()
        real_curvatures = real_data[i].calculate_curvatures_list()[1]
        sim_curvatures = sim_data[i].calculate_curvatures_list()[1]

        x_plot_n = xs[i] - width/2
        x_plot_p = xs[i] + width/2
        ee = width / 7
        x_n = (xs[i] - width + ee)* np.ones(len(sim_distances)) + ee * np.arange(len(sim_distances))
        x_p = (xs[i] +ee )*  np.ones(len(sim_distances)) + ee * np.arange(len(sim_distances))
        print(f"Sim: {np.mean(sim_distances)} -> {sim_distances}")
        a1.bar(x_plot_n, np.mean(sim_distances), color=color_pallet[i], width=width, alpha=0.3, label="Sim")
        a1.bar(x_plot_p, np.mean(real_distances), color=color_pallet[i], width=width, label="Real", hatch='//', alpha=0.5)
        a1.plot(x_n, sim_distances, 'o', color=color_pallet[i])
        a1.plot(x_p, real_distances, 'o', color=color_pallet[i])

        a2.bar(x_plot_n, np.mean(sim_curvatures), color=color_pallet[i], width=width, alpha=0.3, label="Sim")
        a2.bar(x_plot_p, np.mean(real_curvatures), color=color_pallet[i], width=width, label="Real", hatch='//', alpha=0.5)
        a2.plot(x_n, sim_curvatures, 'o', color=color_pallet[i])
        a2.plot(x_p, real_curvatures, 'o', color=color_pallet[i])

    a1.grid(True)
    # a1.set_ylabel("Distance (m)")
    a1.set_title("Total Distance (m)", fontsize=10)
    a1.set_xticks(xs, labels, fontsize=7)
    a1.set_ylim(20, 26)
    a2.grid(True)
    # a2.set_ylabel("Curvature (rad/m)")
    a1.yaxis.set_tick_params(labelsize=8)
    a2.yaxis.set_tick_params(labelsize=8)

    h, l = a1.get_legend_handles_labels()
    fig.legend(h[0:2], l[0:2], loc='lower center', bbox_to_anchor=(0.5, 0.93), ncol=2, fontsize=8)

    # set the y-axis label fontsize

    a2.set_title("Mean curvature (rad/m)", fontsize=10)
    a2.set_xticks(xs, labels, fontsize=7)

    name = "Sim2Real/Imgs/DistanceCurvaturePlot"
    std_img_saving(name)



make_distance_curvature_plot()

