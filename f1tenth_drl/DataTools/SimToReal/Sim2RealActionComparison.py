import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from f1tenth_drl.Planners.TrackLine import TrackLine 
import csv
from matplotlib.ticker import MultipleLocator

from f1tenth_drl.DataTools.MapData import MapData
from f1tenth_drl.DataTools.plotting_utils import *
plt.rcParams['pdf.use14corefonts'] = True

root = "/home/benjy/sim_ws/src/f1tenth_racing/Data/"

class TestLapData:
    def __init__(self, folder, run_number) -> None:
        self.map_name = "CornerHall"
        self.std_track = TrackLine(self.map_name)
        self.std_track.load_test_centerline()

        with open(folder + f"/Run_{run_number}/Run_{run_number}_actions.csv") as file:
            action_reader = csv.reader(file, delimiter=',')
            action_list = []
            for row in action_reader:
                action_list.append(row)
        self.actions = np.array(action_list[1:]).astype(float)
    
        with open(folder + f"/Run_{run_number}/Run_{run_number}_states.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
        self.states = np.array(state_list[1:]).astype(float)

    def extract_xs(self):
        pts = self.states[:, 0:2]
        progresses = [0]
        for pt in pts:
            p = self.std_track.calculate_progress_percent(pt)
            progresses.append(p)
            
        return np.array(progresses[:-1]) * 100



def make_sim2real_steering_overlays():
    agent_names = ["AgentOff_SAC_Game_mco_TAL_8_1_0", "AgentOff_SAC_TrajectoryFollower_mco_TAL_8_1_0", "AgentOff_SAC_endToEnd_mco_TAL_8_1_0", "PurePursuit"]
    labels = ["Full\nplanning", "Trajectory\ntracking", "End-to-end", "Classic"]
    real_runs = [0, 1, 1, 0]
    sim_runs = [0, 0, 0, 0]
    real_folder = "ResultsJetson24/"
    sim_folder = "ResultsRos24/"

    real_data = [TestLapData(root + real_folder + f"{agent_names[i]}", real_runs[i]) for i in range(4)]
    sim_data = [TestLapData(root + sim_folder + f"{agent_names[i]}", sim_runs[i]) for i in range(4)]

    # plt.figure(1, figsize=(5, 2))
    fig, axes = plt.subplots(2, 1, figsize=(5, 3), sharex=True)
    for i in range(4):        
        xs = sim_data[i].extract_xs()
        ys = sim_data[i].actions[:, 0]
        axes[0].plot(xs, ys, color=color_pallet[i], label=labels[i])
        xs = real_data[i].extract_xs()
        ys = real_data[i].actions[:, 0]
        axes[1].plot(xs, ys, color=color_pallet[i])

    axes[0].grid(True)
    plt.xlabel("Track progress (%)", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    axes[0].tick_params(labelsize=8)
    axes[0].text(3, 0.3, "Simulation", fontsize=8, horizontalalignment="left", fontdict={'fontweight': 'bold'})
    axes[1].text(3, 0.3, "Physical Vehicle", fontsize=8, horizontalalignment="left", fontdict={'fontweight': 'bold'})

    axes[0].set_ylabel("Steering Angle (rad)", fontsize=8)
    axes[1].set_ylabel("Steering Angle (rad)", fontsize=8)
    # axes[0].get_yaxis().set_major_locator(MultipleLocator(0.2))
    # plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
    axes[0].set_yticklabels([-0.4, -0.2, 0, 0.2, 0.4])
    axes[1].set_yticklabels([-0.4, -0.2, 0, 0.2, 0.4])
    axes[0].set_ylim(-0.45, 0.45)
    axes[1].set_ylim(-0.45, 0.45)
    plt.xlim(0, 100)

    fig.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.55, 0.93))

    name = "Sim2Real/Imgs/SteeringSim2RealOverlays2"
    std_img_saving(name)

def make_sim2real_speed_overlays():
    agent_names = ["AgentOff_SAC_Game_mco_TAL_8_1_0", "AgentOff_SAC_endToEnd_mco_TAL_8_1_0", "PurePursuit"]
    labels = ["Full planning", "End-to-end", "Classic"]
    real_runs = [4, 1, 1]
    sim_runs = [0, 0, 0]
    real_folder = "ResultsJetson24_2/"
    sim_folder = "ResultsRos25_2/"

    real_data = [TestLapData(root + real_folder + f"{agent_names[i]}", real_runs[i]) for i in range(3)]
    sim_data = [TestLapData(root + sim_folder + f"{agent_names[i]}", sim_runs[i]) for i in range(3)]

    fig, axes = plt.subplots(2, 1, figsize=(5, 2.3), sharex=True)
    color_inds = [0, 2, 3]
    for i in range(3):        
        xs = sim_data[i].extract_xs()
        ys = sim_data[i].actions[:, 1]
        axes[0].plot(xs, ys, color=color_pallet[color_inds[i]], label=labels[i])
        xs = real_data[i].extract_xs()
        ys = real_data[i].actions[:, 1]
        axes[1].plot(xs, ys, color=color_pallet[color_inds[i]])

    axes[0].grid(True)
    plt.xlabel("Track progress (%)", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    axes[0].tick_params(labelsize=8)
    axes[0].text(3, 0.5, "Simulation", fontsize=8, horizontalalignment="left", fontdict={'fontweight': 'bold'},bbox=dict(facecolor='white', alpha=0.6, edgecolor='white'))
    axes[1].text(3, 0.6, "Physical Vehicle", fontsize=8, horizontalalignment="left", fontdict={'fontweight': 'bold'},bbox=dict(facecolor='white', alpha=0.6, edgecolor='white'))

    axes[0].set_ylabel("Speed (m/s)", fontsize=8)
    axes[1].set_ylabel("Speed (m/s)", fontsize=8)
    axes[0].set_ylim(0, 4.5)
    axes[1].set_ylim(0, 4.5)
    axes[0].get_yaxis().set_major_locator(MultipleLocator(1.))
    axes[1].get_yaxis().set_major_locator(MultipleLocator(1.))
    plt.xlim(0, 100)

    fig.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.55, 0.93))

    name = "Sim2Real/Imgs/SpeedSim2RealOverlays2"
    std_img_saving(name)

make_sim2real_speed_overlays()
plt.clf()
make_sim2real_steering_overlays()