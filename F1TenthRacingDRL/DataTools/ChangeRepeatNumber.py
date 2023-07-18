import numpy as np
import glob
import os


def change_repeat_number(folder, new_number):
    map_list = ["aut", "esp", "gbr", "mco"]
    
    for map_name in map_list:
        testing_folder = folder + f"Testing{map_name.upper()}/"
        laps = glob.glob(testing_folder + "Lap*")
        for lap in laps:
            data = np.load(lap)
            new_name = "_".join(lap.split("_")[:-1]) + f"_{new_number}.npy"
            print(new_name)
            np.save(new_name, data)
            os.remove(lap)

change_repeat_number(f"Data/EndMaps_5/AgentOff_SAC_endToEnd_mco_TAL_8_5_1/", 1)
