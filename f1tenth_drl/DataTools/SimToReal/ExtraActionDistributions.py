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
        # self.map_name = MapData(self.map_name)
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



def make_fast_action_dists():
    agent_names = ["AgentOff_SAC_endToEnd_mco_TAL_8_1_0", "PurePursuit"]
    labels = ["End-to-end", "End-to-end", "Classic", "Classic"]
    labels = ["End-to-end", "Classic"]
    real_runs = [2, 2]
    sim_runs = [1, 1]
    real_folder = "ResultsJetson24_2/"
    sim_folder = "ResultsRos25_2/"

    real_data = [TestLapData(root + real_folder + f"{agent_names[i]}", real_runs[i]) for i in range(2)]
    # sim_data = [TestLapData(root + sim_folder + f"{agent_names[i]}", sim_runs[i]) for i in range(2)]
    # data = []
    # data.append(TestLapData(root + sim_folder + f"{agent_names[0]}", sim_runs[0]))
    # data.append(TestLapData(root + real_folder + f"{agent_names[0]}", real_runs[0]))
    # data.append(TestLapData(root + real_folder + f"{agent_names[1]}", real_runs[1]))
    # data.append(TestLapData(root + sim_folder + f"{agent_names[1]}", sim_runs[1]))

    fig, axes = plt.subplots(1, 2, figsize=(4, 1.8), sharex=True, sharey=True)
    color_inds = [2, 3]
    # color_inds = [2, 2, 3, 3]

    for i in range(2):        
        steerings = real_data[i].actions[:, 0]
        speeds = real_data[i].actions[:, 1]
        axes[i].plot(steerings, speeds, '.', color=color_pallet[color_inds[i]], alpha=0.5)
        
        axes[i].grid(True)
        axes[i].set_title(labels[i], fontsize=10)
        axes[i].xaxis.set_tick_params(labelsize=8)
        axes[i].xaxis.set_major_locator(MultipleLocator(0.3))
        
    axes[0].yaxis.set_major_locator(MultipleLocator(1.5))
    axes[0].yaxis.set_tick_params(labelsize=8)
    axes[0].set_ylabel("Speed (m/s)", fontsize=8)
    axes[0].set_xlabel("Steering (rad)", fontsize=8)
    axes[1].set_xlabel("Steering (rad)", fontsize=8)

    # fig.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.55, 0.93))

    name = "Sim2Real/Imgs/FastActionDists"
    std_img_saving(name)



def make_ee_action_dist():
    agent_name = "AgentOff_SAC_endToEnd_mco_TAL_8_1_0"
    # labels = ["End-to-end"]
    labels = [f"3 m/s", f"4 m/s", f"5 m/s"]
    real_runs = [0, 1, 2]
    real_folder = "ResultsJetson24_2/"

    real_data = [TestLapData(root + real_folder + f"{agent_name}", real_runs[i]) for i in range(len(real_runs))]

    fig, axes = plt.subplots(1, len(real_data), figsize=(5, 1.8), sharex=True, sharey=True)
        
    action_steering_list = [d.actions[:, 0] for d in real_data]
    action_speed_list = [d.actions[:, 1] for d in real_data]
    # action_steering_list = [d.actions[:, 0] for d in sim_data]
    # action_speed_list = [d.actions[:, 1] for d in sim_data]
        
    for i in range(len(real_data)):
        axes[i].plot(action_steering_list[i], action_speed_list[i], '.', color=color_pallet[2], alpha=0.5)
        
        axes[i].set_xlim(-0.45, 0.45)
        axes[i].grid(True)
        axes[i].set_title(labels[i], fontsize=10)
        axes[i].xaxis.set_tick_params(labelsize=8)
        axes[i].xaxis.set_major_locator(MultipleLocator(0.3))
        
    axes[0].yaxis.set_major_locator(MultipleLocator(1.5))
    axes[0].yaxis.set_tick_params(labelsize=8)
    axes[0].set_ylim(0.5, 5.5)
    axes[0].set_ylabel("Speed (m/s)", fontsize=10)
    fig.text(0.53, 0.02, "Steering angle (rad)", fontsize=10, ha='center')
    

    name = "Sim2Real/Imgs/EndToEndActionDistribution"
    std_img_saving(name)

def make_action_distributions():
    agent_names = ["AgentOff_SAC_Game_mco_TAL_8_1_0", "AgentOff_SAC_TrajectoryFollower_mco_TAL_8_1_0", "AgentOff_SAC_endToEnd_mco_TAL_8_1_0", "PurePursuit"]
    labels = ["Full\nplanning", "Trajectory\ntracking", "End-to-end", "Classic"]
    real_runs = [1, 0, 0, 1]
    sim_runs = [1, 1, 1, 1]
    real_folder = "ResultsJetson24/"
    sim_folder = "ResultsRos24/"

    real_data = [TestLapData(root + real_folder + f"{agent_names[i]}", real_runs[i]) for i in range(4)]
    sim_data = [TestLapData(root + sim_folder + f"{agent_names[i]}", sim_runs[i]) for i in range(4)]

    
    fig, axes = plt.subplots(1, 4, figsize=(6, 1.8), sharex=True, sharey=True)
        
    action_steering_list = [d.actions[:, 0] for d in real_data]
    action_speed_list = [d.actions[:, 1] for d in real_data]
    # action_steering_list = [d.actions[:, 0] for d in sim_data]
    # action_speed_list = [d.actions[:, 1] for d in sim_data]
        
    for i in range(len(real_data)):
        axes[i].plot(action_steering_list[i], action_speed_list[i], '.', color=color_pallet[i], alpha=0.5)
        
        axes[i].set_xlim(-0.45, 0.45)
        axes[i].grid(True)
        axes[i].set_title(labels[i], fontsize=10)
        axes[i].xaxis.set_tick_params(labelsize=8)
        axes[i].xaxis.set_major_locator(MultipleLocator(0.3))
        
    axes[0].yaxis.set_tick_params(labelsize=8)
    axes[0].set_ylim(1, 3.5)
    axes[0].set_ylabel("Speed (m/s)", fontsize=10)
    fig.text(0.53, 0.02, "Steering angle (rad)", fontsize=10, ha='center')
    

    name = "Sim2Real/Imgs/ActionDistributions2"
    std_img_saving(name)


# make_fast_action_dists()

# make_ee_action_dist()

make_action_distributions()
