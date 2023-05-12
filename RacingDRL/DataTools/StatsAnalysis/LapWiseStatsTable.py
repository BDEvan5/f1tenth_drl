from matplotlib import pyplot as plt
import numpy as np
from RacingDRL.DataTools.plotting_utils import *

def make_lapwise_stats_table():
    map_name = "mco"
    base_path = "Data/"
    set_number = 5
    folder_keys = ["PlanningMaps", "TrajectoryMaps", "EndMaps"]
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]
    folder = base_path + f"LapWise_{set_number}/"
    
    folder_list = [folder + f"AgentOff_SAC_{vehicle_keys[i]}_{map_name}_TAL_8_5_0/" for i in range(3)]
    folder_list.append(folder + f"PurePursuit_PP_pathFollower_{map_name}_test_8_5_0/")
    # folder_list = [base_path + folder_keys[i] + f"_{set_number}/" for i in range(3)]
    
    file_name = base_path + f"Imgs/lapwise_results_table_{set_number}_{map_name.upper()}.txt"
    spacing = 30
    with open(file_name, 'w') as file:
        file.write("\t \\toprule \n")
        file.write("\t  \\textbf{Architecture} &".ljust(spacing))
        file.write("\\textbf{Distance (m)} &".ljust(spacing))
        file.write("\\textbf{Speed (m/s)} &".ljust(spacing))
        file.write("\\textbf{Curvature (rad/cm)} &".ljust(spacing))
        file.write("\\textbf{Lateral deviation (m)} &".ljust(spacing))
        file.write("\\textbf{Speed deviation (m/s)} \\\\ \n".ljust(spacing))
        file.write("\t  \midrule  \n")

    inds = [4, 10, 6, 8]

    for f, folder in enumerate(folder_list):
        read_file = folder + f"DetailSummaryStatistics{map_name.upper()}.txt"
        with open(read_file, 'r') as file:
            lines = file.readlines()
            line_data = lines[4]
            line_data = line_data.split(",")

        with open(file_name, 'a') as write_file:
            if f == 3:
                write_file.write("\t  \midrule  \n")
                
            write_file.write(f"\t {labels[f]} ".ljust(30))

            write_file.write(f" & {float(line_data[3]):.2f}  ".ljust(30))
            for ind in inds:
                write_file.write(f" & {float(line_data[ind]):.2f} $\pm$ {float(line_data[ind+1]):.2f} ".ljust(30))
                  
            write_file.write(f"\\\\ \n")
            
    
    with open(file_name, 'a') as file:
        file.write("\t \\bottomrule")

make_lapwise_stats_table()