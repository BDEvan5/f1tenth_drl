from matplotlib import pyplot as plt
plt.rcParams['pdf.use14corefonts'] = True

import numpy as np
import glob
import os
import math, cmath

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
        
        self.track_progresses = None

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

        if not os.path.exists(self.path + "TrajectoryAnalysis/"): 
            os.mkdir(self.path + "TrajectoryAnalysis/")    
        for self.lap_n in range(5):
            if not self.load_lap_data(): break # no more laps
            self.calculate_state_progress()
            
            self.plot_velocity_heat_map()
            self.plot_tracking_accuracy()
            self.plot_speed_graph()
            self.plot_slip_graph()
            self.plot_steering_graph()

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
    
    def calculate_state_progress(self):
        progresses = []
        for i in range(len(self.states)):
            p = self.std_track.calculate_progress_percent(self.states[i, 0:2])
            progresses.append(p)
            
        self.track_progresses = np.array(progresses)

    def plot_tracking_accuracy(self):
        pts = self.states[:, 0:2]
        thetas = self.states[:, 4]
        racing_cross_track = []
        racing_heading_error = []
        for i in range(len(pts)):
            track_heading, deviation = self.racing_track.get_cross_track_heading(pts[i])
            racing_cross_track.append(deviation)
            heading_error = sub_angles_complex(track_heading, thetas[i])
            # print(f"Track heading: {track_heading}, Vehicle heading: {thetas[i]}, Heading error: {heading_error}")
            racing_heading_error.append(heading_error)
            
        plt.figure(1, figsize=(10, 5))
        plt.clf()
        plt.plot(self.track_progresses, racing_cross_track)
        
        plt.title("Tracking Accuracy (m)")
        plt.xlabel("Track Progress (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.path + f"TrajectoryAnalysis/{self.vehicle_name}_cross_tracking_accuracy_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)
        
        plt.figure(1, figsize=(10, 5))
        plt.clf()
        plt.plot(self.track_progresses, racing_heading_error)
        
        plt.title("Heading Error (rad)")
        plt.xlabel("Track Progress (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.path + f"TrajectoryAnalysis/{self.vehicle_name}_heading_error_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)
        
    def plot_speed_graph(self):
        plt.figure(1, figsize=(10, 5))
        plt.clf()
        plt.plot(self.track_progresses[:-1], self.states[:-1, 3], label="State")
        plt.plot(self.track_progresses[:-1], self.actions[:-1, 1], label="Actions")
    
        plt.legend()
        plt.title("Speed (m/s)")
        plt.xlabel("Track Progress (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.path + f"TrajectoryAnalysis/{self.vehicle_name}_speed_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)  
        
    def plot_steering_graph(self):
        plt.figure(1, figsize=(10, 5))
        plt.clf()
        plt.plot(self.track_progresses[:-1], self.states[:-1, 2], label="State")
        plt.plot(self.track_progresses[:-1], self.actions[:-1, 0], label="Actions")
    
        plt.legend()
        plt.title("Steering (rad)")
        plt.xlabel("Track Progress (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.path + f"TrajectoryAnalysis/{self.vehicle_name}_steering_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)    
        
    def plot_slip_graph(self):
        plt.figure(1, figsize=(10, 5))
        plt.clf()
        plt.plot(self.track_progresses[:-1], self.states[:-1, 6], label="State")
    
        plt.legend()
        plt.title("Slip (rad)")
        plt.xlabel("Track Progress (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.path + f"TrajectoryAnalysis/{self.vehicle_name}_slip_{self.lap_n}.svg", bbox_inches='tight', pad_inches=0)
    
    def plot_velocity_heat_map(self): 
        save_path  = self.path + "TrajectoryAnalysis/"
        
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
        # std_img_saving(name)
        plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)

def sub_angles_complex(a1, a2): 
    real = math.cos(a1) * math.cos(a2) + math.sin(a1) * math.sin(a2)
    im = - math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase

def esp_left_limits():
    plt.xlim(20, 620)
    plt.ylim(50, 520)

def esp_right_limits():
    plt.xlim(900, 1500)
    plt.ylim(50, 520)

def analyse_folder():

    p = "Data/"

    path = p + "testPP_13/"
    # path = p + "testPP_12/"
    # path = p + "main_22"
    
    TestData = AnalyseTestLapData()
    TestData.explore_folder(path)


if __name__ == '__main__':
    analyse_folder()
