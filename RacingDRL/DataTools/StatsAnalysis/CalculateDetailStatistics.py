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
from RacingDRL.Planners.TrackLine import TrackLine
from RacingDRL.Utils.utils import *
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
        stats_file_name = None

    def explore_folder(self, path):
        vehicle_folders = glob.glob(f"{path}*/")
        print(vehicle_folders)
        print(f"{len(vehicle_folders)} folders found")

        for j, self.path in enumerate(vehicle_folders):
            print(f"Vehicle folder being opened: {self.path}")
            
            self.vehicle_name = self.path.split("/")[-2]
            self.map_name = self.vehicle_name.split("_")[3]
            self.process_folder()
            # self.process_all_maps()

    def process_folder(self):
        spacing = 16
        self.stats_file_name = self.path + f"DetailStatistics{self.map_name.upper()}.txt"
        with open(self.stats_file_name, "w") as file:
            file.write(f"Name: {self.path}\n")
            file.write(f"Lap")
            file.write(f"Time".rjust(spacing))
            file.write(f"Progress".rjust(spacing))
            file.write(f"Total Distance".rjust(spacing))
            file.write(f"Avg. Velocity".rjust(spacing))
            file.write(f"Std. Velocity".rjust(spacing))
            file.write(f"Avg. LateralD".rjust(spacing))
            file.write(f"Std. LateralD".rjust(spacing))
            file.write(f"Avg. SpeedD".rjust(spacing))
            file.write(f"Std. SpeedD".rjust(spacing))
            file.write(f"Avg. Curvature".rjust(spacing))
            file.write(f"Std. Curvature".rjust(spacing))
            
            file.write(f" \n")

        self.map_data = MapData(self.map_name)
        self.std_track = TrackLine(self.map_name, False)
        self.racing_track = TrackLine(self.map_name, True)

        for self.lap_n in range(100):
            if not self.load_lap_data(): break # no more laps
            self.calculate_lap_statistics()

        self.generate_summary_stats()
        
    def process_all_maps(self):
        map_names = ["aut", "esp", "gbr", "mco"]
        for map_name in map_names:
            self.map_name = map_name   
            self.process_folder()

    def load_lap_data(self):
        try:
            data = np.load(self.path + f"Testing{self.map_name.upper()}/Lap_{self.lap_n}_history_{self.vehicle_name}.npy")
        except Exception as e:
            print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            return 0
        self.states = data[:, :7]
        self.actions = data[:, 7:]

        return 1 # to say success

    def calculate_lap_statistics(self):
        if not self.load_lap_data(): return

        pts = self.states[:, 0:2]
        ss = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_distance = np.sum(ss)

        time = len(pts) /10

        progress_distance = self.std_track.calculate_progress(pts[-1])
        progress = progress_distance/self.std_track.total_s
        if progress_distance < 0.01 or progress > 0.99:
            progress = 1 # it is finished

        vs = self.states[:, 3]
        avg_velocity = np.mean(vs)
        std_velocity = np.std(vs)
        
        hs = []
        speed_deviations = []
        raceline_speeds = []
        for i, point in enumerate(pts):
            idx, dists = self.racing_track.get_trackline_segment(point)
            x, h = self.racing_track.interp_pts(idx, dists)
            hs.append(h)
            speed = self.racing_track.get_raceline_speed(point)
            deviation = np.abs(speed - vs[i])
            raceline_speeds.append(speed)
            speed_deviations.append(deviation)

        hs = np.array(hs)
        avg_race_deviation = np.mean(hs)
        std_race_deviation = np.std(hs)
        avg_speed_deviation = np.mean(speed_deviations)
        std_speed_deviation = np.std(speed_deviations)
        
        # plt.figure()
        # plt.clf()
        # plt.plot(vs, label="Vehicle Speed")
        # plt.plot(raceline_speeds, label="Raceline Speed")
        # plt.plot(speed_deviations, label="Speed Deviation")
        # plt.legend()
        # plt.title(f"{self.vehicle_name} --> {self.lap_n}")
        # plt.show()
        
        new_xs = np.arange(0, total_distance, 0.2) # 20cm between pts
        cs = np.cumsum(ss)
        cs = np.insert(cs, 0, 0)
        normal_pts = interp_2d_points(new_xs, cs, pts)
        ths, ks = tph.calc_head_curv_num.calc_head_curv_num(normal_pts, new_xs[1:], False)
        ks *= 1000 # to make it per cm
        # ths, ks = tph.calc_head_curv_num.calc_head_curv_num(pts, ss, False)
        mean_curvature = np.mean(np.abs(ks))
        std_curvature = np.std(np.abs(ks))
        
        with open(self.stats_file_name, "a") as file:
            file.write(f"{self.lap_n},")
            file.write(f"{time:14.4f},")
            file.write(f"{progress:14.4f},")
            file.write(f"{total_distance:14.4f},")
            file.write(f"{avg_velocity:14.4f},")
            file.write(f"{std_velocity:14.4f},")
            file.write(f"{avg_race_deviation:14.4f},")
            file.write(f"{std_race_deviation:14.4f},")        
            file.write(f"{avg_speed_deviation:14.4f},")
            file.write(f"{std_speed_deviation:14.4f},")    
            file.write(f"{mean_curvature:14.4f},")
            file.write(f"{std_curvature:14.4f},")

            file.write(f" \n")

    def generate_summary_stats(self):
        progress_ind = 2
        n_values = 12
        mean_data = []
        for i in range(n_values): 
            mean_data.append([])

        n_success, n_total = 0, 0
        progresses = []
        with open(self.stats_file_name, 'r') as file:
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
                        mean_data[i].append(float(line[i]))
                else:
                    continue
        
        progresses = np.array(progresses)
        mean_data = np.array(mean_data)
        with open(self.path + f"DetailSummaryStatistics{self.map_name[-3:].upper()}.txt", "w") as file:
            file.write(lines[0])
            file.write("".join(["-"]*17*n_values) + "\n" )
            file.write(lines[1][:-1] + "  Avg. Progress\n")
            file.write("".join(["-"]*17*n_values) + "\n" )
            file.write("Mean")
            for i in range(1, n_values):
                if i == progress_ind:
                    file.write(f", {np.mean(progresses*100):14.4f}")
                else:
                    avg = np.mean(mean_data[i])
                    file.write(f", {avg:14.4f}")
            file.write(f",           {n_success/n_total * 100}")
            file.write("\n")
            
            file.write("Std ")
            for i in range(1, n_values):
                if i == progress_ind:
                    file.write(f", {np.std(progresses*100):14.4f}")
                else:
                    avg = np.std(mean_data[i])
                    file.write(f", {avg:14.4f}")
            file.write(f",           {n_success/n_total * 100}")
            file.write("\n")
            file.write("".join(["-"]*17*n_values) + "\n" )
            

def interp_2d_points(ss, xp, points):
    xs = np.interp(ss, xp, points[:, 0])
    ys = np.interp(ss, xp, points[:, 1])
    new_pts = np.vstack((xs, ys)).T
    
    return new_pts


def generate_folder_statistics(folder):
    TestData = AnalyseTestLapData()
    TestData.explore_folder(folder)


def analyse_folder():
    p = "Data/"
    
    # path = p + "TrajectoryMaps_8/"
    # path = p + "PlanningMaps_8/"
    # path = p + "LapWise_5/"
    # path = p + "EndMaps_5/"
    path = p + "PurePursuitMaps_5/"
    
    TestData = AnalyseTestLapData()
    TestData.explore_folder(path)

def analyse_all_data():
    p = "Data/"
    set_n = 5
    
    TestData = AnalyseTestLapData()

    path = p + f"PlanningMaps_{set_n}/"
    TestData.explore_folder(path)

    path = p + f"TrajectoryMaps_{set_n}/"
    TestData.explore_folder(path)

    path = p + f"EndMaps_{set_n}/"
    TestData.explore_folder(path)

if __name__ == '__main__':
    analyse_folder()
    # analyse_all_data()
