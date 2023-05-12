from matplotlib import pyplot as plt
plt.rcParams['pdf.use14corefonts'] = True

import numpy as np
import glob
import os

import glob
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import LineCollection

from RacingDRL.DataTools.MapData import MapData
from RacingDRL.Planners.TrackLine import TrackLine 
from RacingDRL.Utils.utils import *
from RacingDRL.DataTools.plotting_utils import *
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

        set = 1
        for j, folder in enumerate(vehicle_folders):
            print(f"Vehicle folder being opened: {folder}")
                
            self.process_folder(folder)

    def process_folder(self, folder):
        self.path = folder

        self.vehicle_name = self.path.split("/")[-2]
        print(f"Vehicle name: {self.vehicle_name}")
        self.map_name = self.vehicle_name.split("_")[3]
        self.map_data = MapData(self.map_name)
        self.std_track = TrackLine(self.map_name, False)
        self.racing_track = TrackLine(self.map_name, True)

        if not os.path.exists(self.path + "TestingVelocities/"): 
            os.mkdir(self.path + "TestingVelocities/")    
        for self.lap_n in range(5):
            if not self.load_lap_data(): break # no more laps
            self.plot_velocity_heat_map()


    def load_lap_data(self):
        try:
            data = np.load(self.path + f"Testing{self.map_name.upper()}/Lap_{self.lap_n}_history_{self.vehicle_name}.npy")
            # data = np.load(self.path + f"Testing/Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
        except Exception as e:
            print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            return 0
        self.states = data[:, :7]
        self.actions = data[:, 7:]

        return 1 # to say success

    
    def plot_velocity_heat_map(self): 
        save_path  = self.path + "TestingVelocities/"
        
        plt.figure(1)
        plt.clf()
        points = self.states[:, 0:2]
        vs = self.states[:, 3]
        
        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(points)
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(5)
        line = plt.gca().add_collection(lc)
        cbar = plt.colorbar(line,fraction=0.046, pad=0.04, shrink=0.99)
        cbar.ax.tick_params(labelsize=25)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        name = save_path + f"{self.vehicle_name}_velocity_map_{self.lap_n}"
        std_img_saving(name)


def esp_left_limits():
    plt.xlim(20, 620)
    plt.ylim(50, 520)

def esp_right_limits():
    plt.xlim(900, 1500)
    plt.ylim(50, 520)

def analyse_folder():

    p = "Data/"

    # path = p + "PurePursuitMaps_1/"
    # path = p + "PlanningMaps_4/"
    # path = p + "EndMaps_2/"
    path = p + "LapWise_5/"
    # path = p + "EndMaps_5/"
    # path = p + "TrajectoryMaps_4/"
    
    TestData = AnalyseTestLapData()
    TestData.explore_folder(path)

def generate_comparative_analysis():
    p = "Data/"
    set_n = 5
    # set_n = "GBR"
    # set_n = "MCO"
    
    # path = p + f"ComparativeAnalysis_{set_n}/"
    # TestData = AnalyseTestLapData()
    # TestData.explore_folder(path)
    
    # path = p + f"TrajectoryMaps_{set_n}/"
    # TestData = AnalyseTestLapData()
    # TestData.explore_folder(path)

    # path = p + f"PurePursuitMaps_{set_n}/"
    # TestData = AnalyseTestLapData()
    # TestData.explore_folder(path)
    
    # path = p + f"PlanningMaps_{set_n}/"
    # TestData = AnalyseTestLapData()
    # TestData.explore_folder(path)
    
    path = p + f"EndMaps_{set_n}/"
    TestData = AnalyseTestLapData()
    TestData.explore_folder(path)


if __name__ == '__main__':
    analyse_folder()
    # generate_comparative_analysis()
    
