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
        racing_track = TrackLine(testing_map.lower(), True)

        for i in range(test_laps):
            states, actions = load_agent_test_data(agent_path + f"Testing{testing_map.upper()}/", vehicle_name, i)

            if states is None: break

            time = len(states) /10
            ss = np.linalg.norm(np.diff(states[:, 0:2], axis=0), axis=1)
            total_distance = np.sum(ss)

            avg_velocity = np.mean(states[:, 3])

            hs = []
            speed_deviations = []
            raceline_speeds = []
            for i, point in enumerate(states[:, 0:2]):
                idx, dists = racing_track.get_trackline_segment(point)
                x, h = racing_track.interp_pts(idx, dists)
                hs.append(h)
                speed = racing_track.get_raceline_speed(point)
                deviation = np.abs(speed - states[i, 3])
                raceline_speeds.append(speed)
                speed_deviations.append(deviation)

            hs = np.array(hs)
            avg_race_deviation = np.mean(hs)
            race_deviation_q1 = np.percentile(hs, 25)
            race_deviation_q3 = np.percentile(hs, 75)
            avg_speed_deviation = np.mean(speed_deviations)
            speed_deviation_q1 = np.percentile(speed_deviations, 25)
            speed_deviation_q3 = np.percentile(speed_deviations, 75)

            progress = std_track.calculate_progress_percent(states[-1, :2])
            if progress < 0.02 or progress > 0.98: # due to occasional calculation errors
                if total_distance < std_track.total_s * 0.8:
                    print(f"Turned around.....")
                    progress = total_distance / std_track.total_s
                    continue
                progress = 1 # it is finished

            agent_data.append({"Lap": i, "TestMap": testing_map, "Distance": total_distance, "Progress": progress, "Time": time, "MeanVelocity": avg_velocity, "LateralD_M": avg_race_deviation, "LateralD_Q1": race_deviation_q1, "LateralD_Q3": race_deviation_q3, "SpeedD_M": avg_speed_deviation, "SpeedD_Q1": speed_deviation_q1, "SpeedD_Q3": speed_deviation_q3})

    agent_df = pd.DataFrame(agent_data)
    agent_df.to_csv(agent_path + "ExtraAgentData.csv", index=False)



def main():
    p = "Data/"
    
    set_number = 1
    # path = p + f"FinalExperiment_{set_number}/"
    path = p + f"Experiment_{set_number}/"

    vehicle_folders = glob.glob(f"{path}*/")
    print(f"{len(vehicle_folders)} folders found")

    for j, path in enumerate(vehicle_folders):
        if path.split("/")[-2] == "Imgs": continue

        create_main_agent_df(path)


if __name__ == '__main__':
    main()



