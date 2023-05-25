import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from RacingDRL.Planners.TrackLine import TrackLine 
import csv
from matplotlib.ticker import MultipleLocator

from RacingDRL.DataTools.MapData import MapData
from RacingDRL.DataTools.plotting_utils import *
plt.rcParams['pdf.use14corefonts'] = True

root = "/home/benjy/sim_ws/src/f1tenth_racing/Data/"


def make_action_overlays():
    agent_names = ["AgentOff_SAC_Game_mco_TAL_8_1_0", "AgentOff_SAC_TrajectoryFollower_mco_TAL_8_1_0", "AgentOff_SAC_endToEnd_mco_TAL_8_1_0", "PurePursuit"]
    labels = ["Full\n planning", "Trajectory\n tracking", "End-to-end", "Classic"]
    real_runs = [0, 1, 1, 0]
    sim_runs = [0, 0, 0, 0]
    real_folder = "ResultsJetson24/"
    sim_folder = "ResultsRos24/"

    real_actions = []
    sim_actions = []

    for i in range(4):
        folder = root + real_folder + f"{agent_names[i]}/Run_{real_runs[i]}"
        with open(folder + f"/Run_{real_runs[i]}_actions.csv") as file:
            action_reader = csv.reader(file, delimiter=',')
            action_list = []
            for row in action_reader:
                action_list.append(row)
            actions = np.array(action_list[1:]).astype(float)
            real_actions.append(actions)

        folder = root + sim_folder + f"{agent_names[i]}/Run_{sim_runs[i]}"
        with open(folder + f"/Run_{sim_runs[i]}_actions.csv") as file:
            action_reader = csv.reader(file, delimiter=',')
            action_list = []
            for row in action_reader:
                action_list.append(row)
            actions = np.array(action_list[1:]).astype(float)
            sim_actions.append(actions)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    for i in range(4):        
        axes[0].plot(real_actions[i][:, 0], color=color_pallet[i], label=labels[i])
        axes[1].plot(sim_actions[i][:, 1], color=color_pallet[i])


    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Speed (m/s)")
    axes[0].set_ylabel("Steering Angle (rad)")
    axes[0].set_ylim(-0.5, 0.5)
    axes[1].grid(True)
    axes[0].grid(True)

    plt.legend(ncol=4)

    name = "Sim2Real/Imgs/ActionOverlays2"
    std_img_saving(name)

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



def make_steering_overlays():
    agent_names = ["AgentOff_SAC_Game_mco_TAL_8_1_0", "AgentOff_SAC_TrajectoryFollower_mco_TAL_8_1_0", "AgentOff_SAC_endToEnd_mco_TAL_8_1_0", "PurePursuit"]
    labels = ["Full\nplanning", "Trajectory\ntracking", "End-to-end", "Classic"]
    real_runs = [0, 1, 1, 0]
    sim_runs = [0, 0, 0, 0]
    real_folder = "ResultsJetson24/"
    sim_folder = "ResultsRos24/"

    real_data = [TestLapData(root + real_folder + f"{agent_names[i]}", real_runs[i]) for i in range(4)]
    sim_data = [TestLapData(root + sim_folder + f"{agent_names[i]}", sim_runs[i]) for i in range(4)]

    plt.figure(1, figsize=(5, 2))
    for i in range(4):        
        xs = sim_data[i].extract_xs()
        ys = sim_data[i].actions[:, 0]
        # xs = real_data[i].extract_xs()
        # ys = real_data[i].actions[:, 0]
        plt.plot(xs, ys, color=color_pallet[i], label=labels[i])


    plt.xlabel("Track progress (%)", fontsize=8)
    # set ytick font size
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel("Steering Angle (rad)", fontsize=8)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.ylim(-0.45, 0.45)
    plt.xlim(0, 100)

    plt.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.99))

    name = "Sim2Real/Imgs/SteeringOverlays2"
    std_img_saving(name)

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

def make_speed_overlays():
    agent_names = ["AgentOff_SAC_Game_mco_TAL_8_1_0", "AgentOff_SAC_TrajectoryFollower_mco_TAL_8_1_0", "AgentOff_SAC_endToEnd_mco_TAL_8_1_0", "PurePursuit"]
    labels = ["Full\n planning", "Trajectory\n tracking", "End-to-end", "Classic"]
    real_runs = [0, 1, 1, 0]
    sim_runs = [1, 1, 1, 1]
    real_folder = "ResultsJetson24/"
    sim_folder = "ResultsRos24/"

    real_actions = []
    sim_actions = []

    for i in range(4):
        folder = root + real_folder + f"{agent_names[i]}/Run_{real_runs[i]}"
        with open(folder + f"/Run_{real_runs[i]}_actions.csv") as file:
            action_reader = csv.reader(file, delimiter=',')
            action_list = []
            for row in action_reader:
                action_list.append(row)
            actions = np.array(action_list[1:]).astype(float)
            real_actions.append(actions)

        folder = root + sim_folder + f"{agent_names[i]}/Run_{sim_runs[i]}"
        with open(folder + f"/Run_{sim_runs[i]}_actions.csv") as file:
            action_reader = csv.reader(file, delimiter=',')
            action_list = []
            for row in action_reader:
                action_list.append(row)
            actions = np.array(action_list[1:]).astype(float)
            sim_actions.append(actions)

    plt.figure(1, figsize=(10, 5))
    for i in range(4):        
        plt.plot(real_actions[i][:, 1], color=color_pallet[i], label=labels[i])


    plt.xlabel("Time (s)")
    plt.ylabel("Steering Angle (rad)")
    # plt.ylim(-0.5, 0.5)
    # plt.xlim(0, 80)

    plt.legend(ncol=4)

    name = "Sim2Real/Imgs/SpeedOverlays3"
    std_img_saving(name)

# make_steering_overlays()
make_sim2real_steering_overlays()
# make_action_overlays()
# make_speed_overlays()
