from matplotlib import pyplot as plt
import numpy as np
from RacingDRL.DataTools.plotting_utils import *


def make_main_results_table():
    train_map = "mco"
    base_path = "Data/"
    set_number = 5
    folder_keys = ["PlanningMaps", "TrajectoryMaps", "EndMaps", "PurePursuitMaps"]

    folder_list = [base_path + folder_keys[i] + f"_{set_number}/" for i in range(4)]
    
    folder_labels = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]

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
        if f < 3:
            means, stds = load_data_mean_std(folder, f"{train_map}_TAL_8_5_testMCO")
        else:
            means, stds = load_data_mean_std(folder, f"{train_map}_test_8_5")

        with open(file_name, 'a') as file:
            
            if f == 3:
                file.write("\t  \midrule  \n")
                
            file.write(f"\t {folder_labels[f]} ".ljust(30))
            for k in keys:
                file.write(f" & {float(means[k][0]):.1f} $\pm$ {float(stds[k][0]):.1f} ".ljust(30))
                
            file.write(f"\\\\ \n")
            
    
    with open(file_name, 'a') as file:
        file.write("\t \\bottomrule")

make_main_results_table()