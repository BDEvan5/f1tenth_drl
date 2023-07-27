from matplotlib import pyplot as plt
import numpy as np
import glob
import os

from PIL import Image
import glob
import trajectory_planning_helpers as tph
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import LineCollection

from F1TenthRacingDRL.DataTools.MapData import MapData
from F1TenthRacingDRL.Planners.TrackLine import TrackLine
from F1TenthRacingDRL.Utils.utils import *
from matplotlib.ticker import MultipleLocator

import pandas as pd

SAVE_PDF = False
# SAVE_PDF = True


def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def load_agent_test_data(test_path, vehicle_name, lap_n):
    file_name = f"Lap_{lap_n}_history_{vehicle_name}.npy"
    try:
        data = np.load(test_path + file_name)
    except Exception as e:
        print(f"No data for: " + file_name)
        return None, None
    
    states = data[:, :7]
    actions = data[:, 7:]

    return states, actions


def create_main_agent_df(agent_path, test_laps=20):
    vehicle_name = agent_path.split("/")[-2]
    train_map = vehicle_name.split("_")[3]

    print(f"Vehicle: {vehicle_name}")

    agent_data = []

    test_folders = glob.glob(agent_path + "Testing*/")
    test_maps = [f.split("/")[-2][-3:] for f in test_folders]
    for testing_map in test_maps:
        std_track = TrackLine(testing_map.lower(), False)

        for i in range(test_laps):
            states, actions = load_agent_test_data(agent_path + f"Testing{testing_map.upper()}/", vehicle_name, i)

            if states is None: break

            time = len(states) /10
            ss = np.linalg.norm(np.diff(states[:, 0:2], axis=0), axis=1)
            total_distance = np.sum(ss)

            avg_velocity = np.mean(states[:, 3])

            progress = std_track.calculate_progress_percent(states[-1, :2])
            if progress < 0.02 or progress > 0.98: # due to occasional calculation errors
                if total_distance < std_track.total_s * 0.8:
                    print(f"Turned around.....")
                    progress = total_distance / std_track.total_s
                    continue
                progress = 1 # it is finished

            agent_data.append({"Lap": i, "TestMap": testing_map, "Distance": total_distance, "Progress": progress, "Time": time, "MeanVelocity": avg_velocity})

    agent_df = pd.DataFrame(agent_data)
    agent_df.to_csv(agent_path + "MainAgentData.csv", index=False)



def main():
    p = "Data/"
    
    set_number = 2
    path = p + f"FinalExperiment_{set_number}/"

    vehicle_folders = glob.glob(f"{path}*/")
    print(f"{len(vehicle_folders)} folders found")

    for j, path in enumerate(vehicle_folders):
        if path.split("/")[-2] == "Imgs": continue

        create_main_agent_df(path)


if __name__ == '__main__':
    main()



