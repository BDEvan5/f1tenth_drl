import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv

from F1TenthRacingDRL.DataTools.MapData import MapData


def find_folders(path="ResultsJetson23"):
    # root_path = "/home/benjy/sim_ws/src/f1tenth_racing/Data/"
    root_path = "/home/benjy/sim_ws/src/F1TenthRacingROS/Data/"
    folders = glob.glob(root_path + path + "/*/Run*")
    print(f"Found {len(folders)} folders")
    return folders


def ensure_path_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

def create_test_paths(folder_name, result_name):
    path = f"Sim2Real/{folder_name}/" 
    ensure_path_exists(path)
    path = f"Sim2Real/{folder_name}/{result_name}/" 
    ensure_path_exists(path)

    return path




def plot_trajectories(folders, folder_name):
    for folder in folders:
        path = create_test_paths(folder_name, "Trajectories")

        name = folder.split("/")[-1]
        number = name.split("_")[-1] 
        print(f"Path: {path} -> {name} -> {number}")

        map_data = MapData("CornerHall") 

        ensure_path_exists(path)

        with open(folder + f"/{name}_states_{number}.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)

        plt.figure(1)
        plt.clf()
        map_data.plot_map_img()
        xs, ys = map_data.xy2rc(states[:, 0], states[:, 1])
        vs = states[:, 3]

        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        max_speed = 6
        norm = plt.Normalize(0, max_speed)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(2)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line, fraction=0.046, pad=0.04, shrink=0.99)

        # plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(path + folder.split("/")[-2] + "__" + name + ".svg")



def plot_actions(folders, folder_name):
    for folder in folders:
        print(f"Analysing folder: {folder}")

        path = create_test_paths(folder_name, "Actions")

        name = folder.split("/")[-1] 
        print(f"Path: {path} -> {name}")

        with open(folder + f"/{name}_actions.csv") as file:
            action_reader = csv.reader(file, delimiter=',')
            action_list = []
            for row in action_reader:
                action_list.append(row)
            actions = np.array(action_list[1:]).astype(float)

        fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        
        axes[0].plot(actions[:, 0], label="Steering")
        axes[0].grid(True)
        axes[0].set_ylabel("Steering Angle (rad)")
        axes[0].set_ylim(-0.5, 0.5)

        axes[1].plot(actions[:, 1], label="Speed")
        axes[1].grid(True)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Speed (m/s)")


        plt.savefig(path + folder.split("/")[-2] + "__" + name + ".svg")


def calculate_states(folders, folder_name):
    for folder in folders:
        path = create_test_paths(folder_name, "Trajectories")

        name = folder.split("/")[-1]
        number = name.split("_")[-1] 
        vehicle = folder.split("/")[-2]
        print(f"Path: {vehicle} -> {name} -> {number}")

        map_data = MapData("CornerHall") 

        ensure_path_exists(path)

        with open(folder + f"/{name}_states_{number}.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)

        pts = states[:, 0:2]
        dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        distances = np.sum(dists)

        ths = states[:, 2]
        dts = ths[1:] - ths[:-1]
        dts = dts[dists > 0.05]
        dists = dists[dists > 0.05]
        curvatures = np.abs(dts / dists)
        total_curvatures = np.sum(curvatures)
        mean_curvatures = np.mean(curvatures)

        print(f"Total distance: {np.sum(distances)}, Total Curvature: {np.sum(total_curvatures)}, Mean Curvature: {np.mean(mean_curvatures)}")


if __name__ == "__main__":
    folder = "ResultsROS"
    folder = "ResultsJetson24_2"
    folder = "ResultsJetson25"
    folder = "SimAug_1"
    folder = "SimAug_1_2"
    folder = "SimAug_1_4"
    folder = "SimAug_1_5"


    fs = find_folders(folder)
    plot_trajectories(fs, folder)
    # plot_actions(fs, folder)

    calculate_states(fs, folder)