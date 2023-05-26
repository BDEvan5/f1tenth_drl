from matplotlib import pyplot as plt
import numpy as np
from RacingDRL.DataTools.plotting_utils import *


def make_single_repeat_table():
    map_name = "mco"
    set_number = 1
    rep_number = 4
    base_path = "Data/"+ f"FinalExperiment_{set_number}/"
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]
    
    folder_list = [base_path + f"AgentOff_SAC_{vehicle_keys[i]}_{map_name}_TAL_8_{set_number}_{rep_number}/" for i in range(3)]
    folder_list.append(base_path + f"PurePursuit_PP_pathFollower_{map_name}_TAL_8_{set_number}_{rep_number}/")
    
    file_name = base_path + f"Imgs/single_repeat_results_table_{set_number}_{map_name.upper()}.txt"
    spacing = 30
    with open(file_name, 'w') as file:
        file.write("\t \\toprule \n")
        file.write("\t  \\textbf{Architecture} &".ljust(spacing))
        file.write("\\textbf{Lap time (s)} &".ljust(spacing))
        file.write("\\textbf{Progress (\%)} \\\\ \n".ljust(spacing))
        # file.write("\\textbf{Distance (m)} &".ljust(spacing))
        file.write("\t  \midrule  \n")

    inds = [1, 2]
    # inds = [1, 2, 3]

    for f, folder in enumerate(folder_list):
        read_file = folder + f"DetailSummaryStatistics{map_name.upper()}.txt"
        with open(read_file, 'r') as file:
            lines = file.readlines()
            mean_data = lines[4].split(",")
            std_data = lines[5].split(",")

        with open(file_name, 'a') as write_file:
            if f == 3:
                write_file.write("\t  \midrule  \n")
                
            write_file.write(f"\t {labels[f]} ".ljust(30))

            for ind in inds:
                write_file.write(f" & {float(mean_data[ind]):.2f} $\pm$ {float(std_data[ind]):.2f} ".ljust(30))
                  
            write_file.write(f"\\\\ \n")
            
    
    with open(file_name, 'a') as file:
        file.write("\t \\bottomrule")

make_single_repeat_table()