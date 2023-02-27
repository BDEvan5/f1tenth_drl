from matplotlib import pyplot as plt
import numpy as np
import glob
import os

from PIL import Image
import glob
import trajectory_planning_helpers as tph
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import LineCollection

from RacingDRL.DataTools.MapData import MapData
from RacingDRL.Planners.StdTrack import StdTrack 
from RacingDRL.Planners.RacingTrack import RacingTrack
from RacingDRL.Utils.utils import *
from matplotlib.ticker import MultipleLocator

# SAVE_PDF = False
SAVE_PDF = True


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
            
            self.process_folder(folder)

    def process_folder(self, folder):
        self.path = folder
        if not os.path.exists(self.path + "TestPaths/"): 
            os.mkdir(self.path + "TestPaths/")    

        self.vehicle_name = self.path.split("/")[-2]
        self.map_name = self.vehicle_name.split("_")[4]
        if self.map_name == "f1":
            self.map_name += "_" + self.vehicle_name.split("_")[5]
        self.map_data = MapData(self.map_name)
        self.std_track = StdTrack(self.map_name)
        self.racing_track = RacingTrack(self.map_name)

        for self.lap_n in range(5):
            if not self.load_lap_data(): break # no more laps
            self.generate_path_plot()


    def load_lap_data(self):
        try:
            data = np.load(self.path + f"Testing/Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
        except Exception as e:
            print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            return 0
        self.states = data[:, :7]
        self.actions = data[:, 7:]

        return 1 # to say success

    def generate_path_plot(self):
        plt.figure(1, figsize=(3, 3))
        plt.clf()

        points = self.states[:, 0:2]        
        self.map_data.plot_map_img()
        self.map_data.plot_centre_line()
        xs, ys = self.map_data.pts2rc(points)
        plt.plot(xs, ys, color='darkorange')

        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # txt = self.vehicle_name.split('_')[1]
        # if len(txt)==3: txt = "Centre line"
        # elif len(txt)==7: txt = "Racing line"
        # plt.text(730, 90, txt, fontsize=11, ha='left', backgroundcolor='white', color='darkblue')

        # plt.xlim(700, 1100)
        # plt.ylim(60, 440)
        plt.tight_layout()
        # plt.show()

        name = self.path + f"TestPaths/{self.vehicle_name}_velocity_map_{self.lap_n}"
        plt.savefig(name + ".svg", bbox_inches='tight')
        if SAVE_PDF:
            plt.savefig(name + ".pdf", bbox_inches='tight', pad_inches=0)




def analyse_folder():
    p = "Data/"

    path = p + "main_22"
    
    TestData = AnalyseTestLapData()
    TestData.explore_folder(path)


if __name__ == '__main__':
    analyse_folder()
