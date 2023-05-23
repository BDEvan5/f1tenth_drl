import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv

from RacingDRL.DataTools.MapData import MapData


def find_folders(path="ResultsJetson23"):
    root_path = "/home/benjy/sim_ws/src/f1tenth_racing/Data/"
    folders = glob.glob(root_path + path + "/*/Run*")
    print(f"Found {len(folders)} folders")
    return folders


def ensure_path_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)



def plot_paths(folders):
    for folder in folders:
        print(f"Analysing folder: {folder}")

        path = "Sim2Real/" + folder.split("/")[-2] + "/"
        name = folder.split("/")[-1]
        print(f"Path: {path} -> {name}")

        map_data = MapData("CornerHall") 

        ensure_path_exists(path)

        with open(folder + f"/{name}_states.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)

        plt.figure(1, figsize=(5, 5))
        plt.clf()
        map_data.plot_map_img()
        xs, ys = map_data.xy2rc(states[:, 0], states[:, 1])
        plt.plot(xs, ys, "k")

        

        # plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(path + name + ".svg")


def plot_trajectories(folders):
    for folder in folders:
        print(f"Analysing folder: {folder}")

        path = "Sim2Real/Trajectories/" 
        name = folder.split("/")[-1] 
        print(f"Path: {path} -> {name}")

        map_data = MapData("CornerHall") 

        ensure_path_exists(path)

        with open(folder + f"/{name}_states.csv") as file:
            state_reader = csv.reader(file, delimiter=',')
            state_list = []
            for row in state_reader:
                state_list.append(row)
            states = np.array(state_list[1:]).astype(float)

        plt.figure(1, figsize=(5, 5))
        plt.clf()
        map_data.plot_map_img()
        xs, ys = map_data.xy2rc(states[:, 0], states[:, 1])
        vs = states[:, 3]

        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 4)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(2)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line, fraction=0.046, pad=0.04, shrink=0.99)

        # plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(path + folder.split("/")[-2] + "__" + name + ".svg")





if __name__ == "__main__":
    fs = find_folders()
    plot_trajectories(fs)

