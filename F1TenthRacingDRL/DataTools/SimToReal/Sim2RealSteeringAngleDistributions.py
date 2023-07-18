import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from F1TenthRacingDRL.Planners.TrackLine import TrackLine 
import csv
from matplotlib.ticker import MultipleLocator

from F1TenthRacingDRL.DataTools.MapData import MapData
from F1TenthRacingDRL.DataTools.plotting_utils import *
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



def make_steering_distributions():
    agent_names = ["AgentOff_SAC_Game_mco_TAL_8_1_0", "AgentOff_SAC_TrajectoryFollower_mco_TAL_8_1_0", "AgentOff_SAC_endToEnd_mco_TAL_8_1_0", "PurePursuit"]
    labels = ["Full\nplanning", "Trajectory\ntracking", "End-to-end", "Classic"]
    real_runs = [0, 1, 1, 0]
    # real_runs = [1, 0, 0, 1]
    # sim_runs = [1, 1, 1, 1]
    sim_runs = [0, 0, 0, 0]
    real_folder = "ResultsJetson24/"
    sim_folder = "ResultsRos24/"

    agent_length = 3
    real_data = [TestLapData(root + real_folder + f"{agent_names[i]}", real_runs[i]) for i in range(agent_length)]
    sim_data = [TestLapData(root + sim_folder + f"{agent_names[i]}", sim_runs[i]) for i in range(agent_length)]

    
    fig, axes = plt.subplots(2, agent_length, figsize=(6, 3), sharex=True, sharey=True)
    # fig, axes = plt.subplots(2, 4, figsize=(6, 1.8), sharex=True, sharey=True)
        
    action_steering_real = [d.actions[:, 0] for d in real_data]
    action_steering_sim = [d.actions[:, 0] for d in sim_data]
        
    for i in range(agent_length):
        data = np.abs(action_steering_sim[i])
        axes[0, i].hist(data, color=color_pallet[i], alpha=0.65, density=True)
        axes[0, i].set_xlim(-0.02, .42)
        axes[0, i].grid(True)
        axes[0, i].set_title(labels[i], fontsize=10)
        axes[0, i].yaxis.set_tick_params(labelsize=8)
       
        data = np.abs(action_steering_real[i])
        axes[1, i].hist(data, color=color_pallet[i], alpha=0.65, density=True)
        axes[1, i].grid(True)
        axes[1, i].xaxis.set_tick_params(labelsize=8)
        axes[1, i].yaxis.set_tick_params(labelsize=8)
        axes[1, i].xaxis.set_major_locator(MultipleLocator(0.1))
        
    axes[0,0].yaxis.set_tick_params(labelsize=8)
    axes[0, 0].set_ylabel("Simulation", fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel("Physical \nvehicle", fontsize=10, fontweight='bold')
    fig.text(0.53, 0.0, "Absolute Steering angle (rad)", fontsize=10, ha='center')

    

    name = "Sim2Real/Imgs/SteeringDistributions2"
    std_img_saving(name)


make_steering_distributions()