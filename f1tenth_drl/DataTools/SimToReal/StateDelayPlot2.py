import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv
import trajectory_planning_helpers as tph

from f1tenth_drl.DataTools.MapData import MapData
from f1tenth_drl.DataTools.plotting_utils import *
from f1tenth_drl.Utils.utils import *

root = "/home/benjy/sim_ws/src/F1TenthRacingROS/Data/"


def interp_2d_points(ss, xp, points):
    xs = np.interp(ss, xp, points[:, 0])
    ys = np.interp(ss, xp, points[:, 1])
    new_pts = np.vstack((xs, ys)).T
    
    return new_pts

def calculate_curvature(points):
    ss = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_distance = np.sum(ss)
    new_xs = np.arange(0, total_distance, 0.2) # 20cm between pts
    cs = np.cumsum(ss)
    cs = np.insert(cs, 0, 0)
    normal_pts = interp_2d_points(new_xs, cs, points)
    ths, ks = tph.calc_head_curv_num.calc_head_curv_num(normal_pts, new_xs[1:], False)
    ks *= 1000 # to make it per cm
    mean_curvature = np.mean(np.abs(ks))
    # std_curvature = np.std(np.abs(ks))
    total_curvature = np.sum(np.abs(ks))

    return mean_curvature, total_curvature


class SpeedResults:
    def __init__(self, agent_name):
        self.agent_name = agent_name

        self.states = []
        self.delays = []
        self.actions = []

    def load_state_data(self, folder, run_n, delay):
        folder = root + folder + f"{self.agent_name}/Run_{run_n}"
        with open(folder + f"/Run_{run_n}_states_{run_n}.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)
        self.states.append(states)
        self.delays.append(delay*0.04*1000 )

        with open(folder + f"/Run_{run_n}_actions_{run_n}.csv") as file:
            action_reader = csv.reader(file, delimiter=',')
            action_list = []
            for row in action_reader:
                action_list.append(row)
            actions = np.array(action_list[1:]).astype(float)
        self.actions.append(actions)


    def get_lap_times(self):
        lap_times = []
        for i in range(len(self.states)):
            lap_times.append(len(self.states[i])*0.1)

        return self.delays, lap_times
    
    def get_curvatures(self):

        curvatures = []
        for i in range(len(self.states)):
            points = self.states[i][:, :2] 
            mean_c, total_c = calculate_curvature(points)
            curvatures.append(total_c)

        return self.delays, curvatures
    
    def get_steers(self):

        steer_list = []
        for i in range(len(self.states)):
            steers = self.actions[i][:, 0] 
            total = np.sum(np.abs(steers))
            steer_list.append(total)

        return self.delays, steer_list
    
    def calculate_distance_list(self):
        distances = []
        for i in range(len(self.states)):
            pts = self.states[i][:, 0:2]
            dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            distances.append(np.sum(dists))

        return self.delays, distances
    
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

        return self.delays, mean_curvatures
        # return total_curvatures, mean_curvatures


def make_fast_laptime_plot():
    agent_names = ["AgentOff_SAC_Game_mco_TAL_8_1_0", "AgentOff_SAC_TrajectoryFollower_mco_TAL_8_1_0", "AgentOff_SAC_endToEnd_mco_TAL_8_1_0", "PurePursuit"]
    labels = ["Full\nplanning", "Trajectory\ntracking", "End-to-end", "Classic"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]
    
    sim_folder = "SimAug_2_2/"
    # real_folder23 = "ResultsJetson23/"
    # real_folder24 = "ResultsJetson24/"
    # real_folder242 = "ResultsJetson24_2/"
    # sim_folder = "ResultsRos24/"

    real_data = [SpeedResults(agent_names[i]) for i in range(4)]
    # sim_data = [SpeedResults(agent_names[i]) for i in range(4)]

    real_data[0].load_state_data(sim_folder, 0, 6)
    real_data[0].load_state_data(sim_folder, 1, 4)
    real_data[0].load_state_data(sim_folder, 2, 2)
    real_data[0].load_state_data(sim_folder, 3, 0)
    real_data[0].load_state_data(sim_folder, 6, 8)
    # real_data[0].load_state_data(sim_folder, 5, 10)
    # real_data[0].load_state_data(sim_folder, 5, 5)

    real_data[1].load_state_data(sim_folder, 0, 6)
    real_data[1].load_state_data(sim_folder, 1, 4)
    real_data[1].load_state_data(sim_folder, 2, 2)
    real_data[1].load_state_data(sim_folder, 3, 0)

    real_data[2].load_state_data(sim_folder, 0, 6)
    real_data[2].load_state_data(sim_folder, 1, 4)
    real_data[2].load_state_data(sim_folder, 2, 2)
    real_data[2].load_state_data(sim_folder, 7, 0)
    real_data[2].load_state_data(sim_folder, 4, 8)
    # real_data[2].load_state_data(sim_folder, 5, 10)

    real_data[3].load_state_data(sim_folder, 3, 0)
    real_data[3].load_state_data(sim_folder, 0, 2)
    real_data[3].load_state_data(sim_folder, 1, 4)
    real_data[3].load_state_data(sim_folder, 4, 6)
    real_data[3].load_state_data(sim_folder, 2, 8)

    xs = np.arange(len(labels))
    # width = 0.2 * 100
    width = 17
    fig = plt.figure(figsize=(5.5, 2.))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)


    # plt.figure(figsize=(4.5, 2))
    for i in range(len(real_data)):
        real_speeds, real_times = real_data[i].calculate_distance_list()
        # real_speeds, real_times = real_data[i].calculate_curvatures_list()
        # real_speeds, real_times = real_data[i].calculate_distance_list()
        # real_speeds, real_times = real_data[i].get_curvatures()
        # real_speeds, real_times = real_data[i].get_lap_times()

        x_off = (-width * 1.5 + width * i) * np.ones_like(real_speeds)
        # x_off = (-width * 1.5 + width * i) * np.ones_like(real_speeds)
        # a1.bar(x_plot_n, np.mean(sim_distances), color=color_pallet[i], width=width, alpha=0.3, label="Sim")
        # plt.bar(real_speeds + x_off, real_times, color=color_pallet[i], width=width, label="Real", alpha=0.6)
        # plt.bar(real_speeds + x_off, real_times, color=color_pallet[i], width=width, label="Real", hatch='', alpha=0.5)

        ax1.bar(real_speeds + x_off, real_times, color=color_pallet[i], width=width, label="Real", alpha=0.6)

        real_speeds, real_times = real_data[i].calculate_curvatures_list()
        ax2.bar(real_speeds + x_off, real_times, color=color_pallet[i], width=width, label="Real", alpha=0.6)

    ax1.set_title("Total distance (m)", fontsize=10)
    # ax1.set_xticks(xs, labels, fontsize=7)
    ax1.set_ylim(20.5, 23.5)
    ax1.grid(True, axis='y')
    # a2.set_ylabel("Curvature (rad/m)")
    ax1.yaxis.set_tick_params(labelsize=8)
    ax2.yaxis.set_tick_params(labelsize=8)
    ax2.xaxis.set_tick_params(labelsize=8)
    ax1.xaxis.set_tick_params(labelsize=8)

    ax1.set_xticks([0, 80, 160, 240, 320])
    ax2.set_xticks([0, 80, 160, 240, 320])

    ax2.yaxis.set_major_locator(plt.MaxNLocator(6))

    ax2.grid(True, axis='y')
    # plt.ylim(5, 16)
    ax1.set_xlabel("Localisation delay (ms)")
    ax2.set_xlabel("Localisation delay (ms)")
    # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))

    ax2.set_title("Mean curvature (rad/m)", fontsize=10)
    # ax2.set_xticks(xs, labels, fontsize=7)

    # h, l = plt.gca().get_legend_handles_labels()
    # plt.legend(labels, loc="upper right", ncol=2, fontsize=8)
    fig.legend(labels, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=4, fontsize=8)

    name = "Sim2Real/Imgs/StateDelayPlot2"
    std_img_saving(name)



make_fast_laptime_plot()

