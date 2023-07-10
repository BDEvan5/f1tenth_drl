from matplotlib import pyplot as plt
import numpy as np
import glob
import os

from PIL import Image
import glob
import trajectory_planning_helpers as tph
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import LineCollection

from RacingRewards.DataTools.MapData import MapData
from RacingRewards.RewardSignals.StdTrack import StdTrack 
from RacingRewards.RewardSignals.RacingTrack import RacingTrack
from RacingRewards.Utils.utils import *
from matplotlib.ticker import MultipleLocator

SAVE_PDF = False
# SAVE_PDF = True


def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class AnalyseTestLapData:
    def __init__(self):
        self.path = None
        self.vehicle_name = None
        self.map_name = None
        self.states = None
        self.actions = None
        self.map_data = None
        self.std_track = None
        self.summary_path = None
        self.lap_n = 0

    def explore_folder(self, path):
        vehicle_folders = glob.glob(f"{path}*/")
        print(vehicle_folders)
        print(f"{len(vehicle_folders)} folders found")

        for j, folder in enumerate(vehicle_folders):
            print(f"Vehicle folder being opened: {folder}")
            
            self.process_folder_all_maps(folder)
        
    def process_folder_all_maps(self, folder):
        self.path = folder
        self.vehicle_name = self.path.split("/")[-2]
        if self.vehicle_name == "_Imgs" or self.vehicle_name == "Imgs": return
        
        map_names = ["aut", "esp", "gbr", "mco"]
        spacing = 16
        for self.map_name in map_names:

            self.lap_n = 0
            if not self.load_lap_data(): continue # no data

            with open(self.path + f"Statistics{self.map_name[-3:].upper()}.txt", "w") as file:
                file.write(f"Name: {self.path}\n")
                file.write(f"Lap")
                file.write(f"Total Distance".rjust(spacing))
                file.write(f"Progress".rjust(spacing))
                file.write(f"Time".rjust(spacing))
                file.write(f"Avg. Velocity".rjust(spacing))
                file.write(f" \n")

            self.map_data = MapData(self.map_name)
            self.std_track = StdTrack(self.map_name)
            self.racing_track = RacingTrack(self.map_name)

            for self.lap_n in range(100):
                if not self.load_lap_data(): break # no more laps
                self.calculate_lap_statistics()

            self.generate_summary_stats()

    def load_lap_data(self):
        file_name = f"Testing{self.map_name.upper()}/Lap_{self.lap_n}_history_{self.vehicle_name}.npy"
        try:
            data = np.load(self.path + file_name)
        except Exception as e:
            print(f"No data for: " + file_name)
            return 0
        self.states = data[:, :7]
        self.actions = data[:, 7:]

        return 1 # to say success

    def calculate_lap_statistics(self):
        if not self.load_lap_data(): return

        pts = self.states[:, 0:2]
        ss = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_distance = np.sum(ss)
        if self.map_name == "esp":
            breakpoint

        time = len(pts) /10
        vs = self.states[:, 3]
        avg_velocity = np.mean(vs)

        progress_distance = self.std_track.calculate_progress(pts[-1])
        progress = progress_distance/self.std_track.total_s
        if progress_distance < 0.01 or progress > 0.99:
            progress = 1 # it is finished

        with open(self.path + f"Statistics{self.map_name[-3:].upper()}.txt", "a") as file:
            file.write(f"{self.lap_n},  {total_distance:14.4f},  {progress:14.4f}, {time:14.4f}, {avg_velocity:14.4f} \n")

    def generate_summary_stats(self):
        progress_ind = 2
        n_values = 5
        data = []
        for i in range(n_values): 
            data.append([])

        n_success, n_total = 0, 0
        progresses = []
        with open(self.path + f"Statistics{self.map_name[-3:].upper()}.txt", 'r') as file:
            lines = file.readlines()
            if len(lines) < 3: return
            
            for lap_n in range(len(lines)-2):
                line = lines[lap_n+2] # first lap is heading
                line = line.split(',')
                progress = float(line[progress_ind])
                n_total += 1
                progresses.append(progress)
                if progress < 0.005 or progress > 0.99:
                    n_success += 1
                    for i in range(n_values):
                        data[i].append(float(line[i]))
                else:
                    continue
        
        progresses = np.array(progresses)
        data = np.array(data)
        with open(self.path + f"SummaryStatistics{self.map_name[-3:].upper()}.txt", "w") as file:
            file.write(lines[0])
            file.write("".join(["-"]*17*n_values) + "\n" )
            file.write(lines[1][:-1] + "  Avg. Progress\n")
            file.write("".join(["-"]*17*n_values) + "\n" )
            file.write("0  ")
            for i in range(1, n_values):
                if i == progress_ind:
                    file.write(f", {np.mean(progresses*100):14.4f}")
                else:
                    avg = np.mean(data[i])
                    file.write(f", {avg:14.4f}")
            file.write(f",           {n_success/n_total * 100}")
            file.write("\n")
            file.write("".join(["-"]*17*n_values) + "\n" )
            

def generate_folder_statistics(folder):
    TestData = AnalyseTestLapData()
    TestData.explore_folder(folder)


def analyse_folder():
    p = "Data/"
    
    path = p + "FinalExperiment_1/"
    
    TestData = AnalyseTestLapData()
    TestData.explore_folder(path)


if __name__ == '__main__':
    analyse_folder()
  

  