import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv

from RacingDRL.DataTools.MapData import MapData
from RacingDRL.DataTools.plotting_utils import *

root = "/home/benjy/sim_ws/src/f1tenth_racing/Data/"


class SpeedResults:
    def __init__(self, agent_name):
        self.agent_name = agent_name

        self.states = []
        self.speeds = []

    def load_state_data(self, folder, run_n, speed):
        folder = root + folder + f"{self.agent_name}/Run_{run_n}"
        with open(folder + f"/Run_{run_n}_states.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)
        self.states.append(states)
        self.speeds.append(speed)

    def get_lap_times(self):
        lap_times = []
        for i in range(len(self.states)):
            lap_times.append(len(self.states[i])*0.1)

        return self.speeds, lap_times



def make_distance_curvature_plot():
    agent_names = ["AgentOff_SAC_Game_mco_TAL_8_1_0", "AgentOff_SAC_TrajectoryFollower_mco_TAL_8_1_0", "AgentOff_SAC_endToEnd_mco_TAL_8_1_0", "PurePursuit"]
    labels = ["Full\nplanning", "Trajectory\ntracking", "End-to-end", "Classic"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]
    
    real_folder23 = "ResultsJetson23/"
    real_folder24 = "ResultsJetson24/"
    real_folder242 = "ResultsJetson24_2/"
    sim_folder = "ResultsRos24/"

    real_data = [SpeedResults(agent_names[i]) for i in range(4)]
    sim_data = [SpeedResults(agent_names[i]) for i in range(4)]

    real_data[0].load_state_data(real_folder242, 1, 2)
    real_data[0].load_state_data(real_folder242, 3, 3)
    real_data[0].load_state_data(real_folder242, 4, 4)

    real_data[1].load_state_data(real_folder24, 0, 2)

    real_data[2].load_state_data(real_folder24, 1, 2)
    real_data[2].load_state_data(real_folder242, 0, 3)
    real_data[2].load_state_data(real_folder242, 1, 4)
    real_data[2].load_state_data(real_folder242, 2, 5)

    real_data[3].load_state_data(real_folder24, 1, 2)
    real_data[3].load_state_data(real_folder242, 0, 3)
    real_data[3].load_state_data(real_folder242, 1, 4)
    real_data[3].load_state_data(real_folder242, 2, 5)

    xs = np.arange(len(labels))
    width = 0.18

    plt.figure(figsize=(4.5, 2))
    # plt.figure(figsize=(5.5, 2.2))
    for i in range(len(real_data)):
        real_speeds, real_times = real_data[i].get_lap_times()

        x_off = (-width * 1.5 + width * i) * np.ones_like(real_speeds)
        # a1.bar(x_plot_n, np.mean(sim_distances), color=color_pallet[i], width=width, alpha=0.3, label="Sim")
        plt.bar(real_speeds + x_off, real_times, color=color_pallet[i], width=width, label="Real", alpha=0.6)
        # plt.bar(real_speeds + x_off, real_times, color=color_pallet[i], width=width, label="Real", hatch='', alpha=0.5)

    plt.ylim(5, 16)
    plt.grid(True)
    plt.ylabel("Lap time (s)")
    plt.xlabel("Speed cap (m/s)")
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(3))

    # h, l = plt.gca().get_legend_handles_labels()
    plt.legend(labels, loc="upper right", ncol=2, fontsize=8)

    name = "Sim2Real/Imgs/FastLapTimesPlot"
    std_img_saving(name)



make_distance_curvature_plot()

