import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
from RacingDRL.DataTools.MapData import MapData
from RacingDRL.Planners.TrackLine import TrackLine 

from RacingDRL.DataTools.plotting_utils import *

class TestLapData:
    def __init__(self, path, lap_n=0):
        self.path = path
        
        self.vehicle_name = self.path.split("/")[-2]
        print(f"Vehicle name: {self.vehicle_name}")
        self.map_name = self.vehicle_name.split("_")[3]
        self.map_data = MapData(self.map_name)
        self.std_track = TrackLine(self.map_name, False)
        self.racing_track = TrackLine(self.map_name, True)
        
        self.states = None
        self.actions = None
        self.lap_n = lap_n

        self.load_lap_data()

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

    def generate_state_progress_list(self):
        pts = self.states[:, 0:2]
        progresses = [0]
        for pt in pts:
            p = self.std_track.calculate_progress_percent(pt)
            # if p < progresses[-1]: continue
            progresses.append(p)
            
        return np.array(progresses[:-1]) * 100

    def calculate_deviations(self):
        # progresses = self.generate_state_progress_list()
        pts = self.states[:, 0:2]
        deviations = []
        for pt in pts:
            _, deviation = self.racing_track.get_cross_track_heading(pt)
            deviations.append(deviation)
            
        return np.array(deviations)

    def calculate_curvature(self, n_xs=200):
        thetas = self.states[:, 4]
        xs = np.linspace(0, 100, n_xs)
        theta_p = np.interp(xs, self.generate_state_progress_list(), thetas)
        curvatures = np.diff(theta_p) / np.diff(xs)
        
        return curvatures
    
    

def speed_profile_comparison():
    map_name = "gbr"
    # map_name = "mco"
    base = f"Data/ComparativeAnalysis_{map_name.upper()}/"
    names = ["Classic", "Full planning", "Trajectory tracking", "End-to-end"]
    sub_paths = [f"PurePursuit_PP_pathFollower_{map_name}_test_8_8_0/",
                 f"AgentOff_SAC_Game_{map_name}_train_8_8_0/",
                 f"AgentOff_SAC_TrajectoryFollower_{map_name}_train_8_8_0/",
                 f"AgentOff_SAC_endToEnd_{map_name}_train_8_8_0/"]
    
    data_list = [TestLapData(base + p, 0) for p in sub_paths]
    xs_list = [d.generate_state_progress_list() for d in data_list]
    xs = np.linspace(0, 100, 200)
    speed_list = [np.interp(xs, xs_list[i], data_list[i].states[:, 3]) for i in range(len(data_list))]
    
    plt.figure(1, figsize=(6, 1.9))
    ax1 = plt.gca()
    for n, name in enumerate(names):
        ax1.plot(xs, speed_list[n], color=pp[n], label=name)

    ax1.set_ylabel("Speed m/s")
    ax1.set_xlabel("Track progress (%)")
    ax1.legend(ncol=4, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 0.96))

    plt.grid(True)
    plt.tight_layout()
    plt.xlim(10, 60)

    plt.savefig(f"{base}_Imgs/CompareSpeed_{map_name.upper()}.svg", bbox_inches='tight')
    plt.savefig(f"{base}_Imgs/CompareSpeed_{map_name.upper()}.pdf", bbox_inches='tight')

def curvature_profile_comparison():
    map_name = "gbr"
    # map_name = "mco"
    base = f"Data/ComparativeAnalysis_{map_name.upper()}/"
    names = ["Classic", "Full planning", "Trajectory tracking", "End-to-end"]
    sub_paths = [f"PurePursuit_PP_pathFollower_{map_name}_test_8_8_0/",
                 f"AgentOff_SAC_Game_{map_name}_train_8_8_0/",
                 f"AgentOff_SAC_TrajectoryFollower_{map_name}_train_8_8_0/",
                 f"AgentOff_SAC_endToEnd_{map_name}_train_8_8_0/"]
    
    data_list = [TestLapData(base + p, 0) for p in sub_paths]
    
    xs = np.linspace(0, 100, 200)
    curve_list = [d.calculate_curvature(200) for d in data_list]

    plt.figure(1, figsize=(6, 1.9))
    ax1 = plt.gca()
    for n, name in enumerate(names):
        ax1.plot(xs[:-1], np.abs(curve_list[n]), color=pp[n], label=name)
    
    ax1.set_ylabel("Curvature (rad/m)")
    ax1.set_xlabel("Track progress (%)")
    ax1.legend(ncol=4, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 0.96))

    plt.grid(True)
    plt.tight_layout()
    plt.xlim(10, 60)
    plt.xlim(30, 60)
    plt.gca().yaxis.set_major_locator(MultipleLocator(1.5))
    y_lim = 4
    plt.ylim(0, y_lim)
    # plt.ylim(-y_lim, y_lim)

    plt.savefig(f"{base}_Imgs/CompareCurvature_{map_name.upper()}.svg", bbox_inches='tight')
    plt.savefig(f"{base}_Imgs/CompareCurvature_{map_name.upper()}.pdf", bbox_inches='tight')

    # plt.show()
    
def speed_profile_deviation():
    map_name = "gbr"
    # map_name = "mco"
    base = f"Data/ComparativeAnalysis_{map_name.upper()}/"
    names = ["Classic", "Full planning", "Trajectory tracking", "End-to-end"]
    sub_paths = [f"PurePursuit_PP_pathFollower_{map_name}_test_8_8_0/",
                 f"AgentOff_SAC_Game_{map_name}_train_8_8_0/",
                 f"AgentOff_SAC_TrajectoryFollower_{map_name}_train_8_8_0/",
                 f"AgentOff_SAC_endToEnd_{map_name}_train_8_8_0/"]
    
    data_list = [TestLapData(base + p, 0) for p in sub_paths]
    xs_list = [d.generate_state_progress_list() for d in data_list]
    xs = np.linspace(0, 100, 200)
    speed_list = [np.interp(xs, xs_list[i], data_list[i].states[:, 3]) for i in range(len(data_list))]
    
    plt.figure(1, figsize=(6, 1.9))
    ax1 = plt.gca()
    for n, name in enumerate(names[1:]):
        ax1.plot(xs, speed_list[n+1] - speed_list[0], color=pp[n+1], label=name)

    ax1.set_ylabel("Speed \ndifference m/s", fontsize=10)
    ax1.set_xlabel("Track progress (%)")
    ax1.legend(ncol=4, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 0.96))

    plt.grid(True)
    plt.tight_layout()
    plt.xlim(10, 60)
    plt.gca().yaxis.set_major_locator(MultipleLocator(2.5))

    plt.savefig(f"{base}_Imgs/SpeedDifference_{map_name.upper()}.svg", bbox_inches='tight')
    plt.savefig(f"{base}_Imgs/SpeedDifference_{map_name.upper()}.pdf", bbox_inches='tight')

    # plt.show()
    

def lateral_deviation():
    map_name = "gbr"
    # map_name = "mco"
    base = f"Data/ComparativeAnalysis_{map_name.upper()}/"
    names = ["Classic", "Full planning", "Trajectory tracking", "End-to-end"]
    sub_paths = [f"PurePursuit_PP_pathFollower_{map_name}_test_8_8_0/",
                 f"AgentOff_SAC_Game_{map_name}_train_8_8_0/",
                 f"AgentOff_SAC_TrajectoryFollower_{map_name}_train_8_8_0/",
                 f"AgentOff_SAC_endToEnd_{map_name}_train_8_8_0/"]
    
    data_list = [TestLapData(base + p, 0) for p in sub_paths]
    xs_list = [d.generate_state_progress_list() for d in data_list]
    deviation_list = [d.calculate_deviations() * 100 for d in data_list]
    
    plt.figure(1, figsize=(6, 1.9))
    ax1 = plt.gca()
    for n, name in enumerate(names):
        ax1.plot(xs_list[n], deviation_list[n], color=pp[n], label=name)

    ax1.set_ylabel("Lateral \ndeviation (cm)", fontsize=9)
    ax1.set_xlabel("Track progress (%)")
    ax1.legend(ncol=4, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 0.96))

    plt.grid(True)
    plt.tight_layout()
    plt.xlim(10, 50)
    plt.gca().yaxis.set_major_locator(MultipleLocator(20))

    plt.savefig(f"{base}_Imgs/LateralDeviation_{map_name.upper()}.svg", bbox_inches='tight')
    plt.savefig(f"{base}_Imgs/LateralDeviation_{map_name.upper()}.pdf", bbox_inches='tight')

    # plt.show()


# speed_profile_comparison()
# speed_profile_deviation()
# lateral_deviation()
curvature_profile_comparison()
