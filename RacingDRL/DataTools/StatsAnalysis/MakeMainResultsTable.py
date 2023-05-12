from matplotlib import pyplot as plt
import numpy as np
from RacingDRL.DataTools.plotting_utils import *


def make_main_results_table():
    map_names = ["aut", "esp", "gbr", "mco"]
    train_map = "mco"
    base_path = "Data/"
    set_number = 5
    folder_keys = ["PlanningMaps", "TrajectoryMaps", "EndMaps"]
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end"]

    base_path = "Data/"
    set_number = 5
    test_name = "PlanningMaps" 
    folder_games = base_path + test_name + f"_{set_number}/"
    
    test_name = "TrajectoryMaps" 
    folder_traj = base_path + test_name + f"_{set_number}/"
    
    test_name = "EndMaps" 
    folder_pp = base_path + test_name + f"_{set_number}/"
    
    folder_list = [folder_games, folder_traj, folder_pp]
    folder_labels = ["Full planning", "Trajectory tracking", "End-to-end"]

    file_name = base_path + f"Imgs/main_results_table_{train_map.upper()}.txt"
    spacing = 30
    with open(file_name, 'w') as file:
        file.write("\t \\toprule \n")
        file.write("\t  \\textbf{Architecture} &".ljust(spacing))
        file.write("\\textbf{Lap time (s)} &".ljust(spacing))
        file.write("\\textbf{Success (\%)} &".ljust(spacing))
        file.write("\\textbf{Avg. Progress (\%)} \\\\ \n".ljust(spacing))
        file.write("\t  \midrule \n")

    keys = ["time", "success", "progress"]
    
    for f, folder in enumerate(folder_list):
        means, stds = load_data_mean_std(folder, f"{train_map}_TAL_8_5_testMCO")

        with open(file_name, 'a') as file:
            file.write(f"\t {folder_labels[f]} ".ljust(30))
            for k in keys:
                file.write(f" & {float(means[k][0]):.1f} $\pm$ {float(stds[k][0]):.1f} ".ljust(30))
                
            file.write(f"\\\\ \n")
            
    
    with open(file_name, 'a') as file:
        file.write("\t \\bottomrule")

make_main_results_table()