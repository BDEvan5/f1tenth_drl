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


    
    

def speed_profile_comparison():
    map_name = "mco"
    pp_path = f"Data/ComparativeAnalysis_8/PurePursuit_PP_pathFollower_mco_test_8_8_0/"
    planner_path = f"Data/ComparativeAnalysis_8/AgentOff_SAC_Game_mco_train_8_8_0/"
    trajectory_path = f"Data/ComparativeAnalysis_8/AgentOff_SAC_TrajectoryFollower_mco_train_8_8_0/"
    end_path = f"Data/ComparativeAnalysis_8/AgentOff_SAC_endToEnd_mco_train_8_8_0/"

    pp_data = TestLapData(pp_path)
    planner_data = TestLapData(planner_path, 0)
    trajectory_data = TestLapData(trajectory_path, 0)
    end_data = TestLapData(end_path, 0)

    classic_xs = pp_data.generate_state_progress_list()
    planner_xs = planner_data.generate_state_progress_list()
    trajectory_xs = trajectory_data.generate_state_progress_list()
    end_xs = end_data.generate_state_progress_list()

    # fig, (ax1) = plt.subplots(1, 1, figsize=(6, 1.7), sharex=True)
    plt.figure(1, figsize=(6, 1.9))
    ax1 = plt.gca()
    ax1.plot(planner_xs, planner_data.states[:, 3], color=pp[1], label="Full planner")
    ax1.plot(trajectory_xs, trajectory_data.states[:, 3], color=pp[2], label="Trajectory follower")
    ax1.plot(end_xs, end_data.states[:, 3], color=pp[3], label="End-to-end")
    ax1.plot(classic_xs, pp_data.states[:, 3], color=pp[0], label="Classic")

    ax1.set_ylabel("Speed m/s")
    ax1.set_xlabel("Track progress (%)")
    ax1.legend(ncol=4, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 0.96))

    plt.grid(True)
    plt.tight_layout()
    plt.xlim(10, 60)

    plt.savefig(f"Data/ComparativeAnalysis_8/_Imgs/CompareSpeed_{map_name.upper()}.svg", bbox_inches='tight')
    plt.savefig(f"Data/ComparativeAnalysis_8/_Imgs/CompareSpeed_{map_name.upper()}.pdf", bbox_inches='tight')

def curvature_profile_comparison():
    map_name = "mco"
    pp_path = f"Data/ComparativeAnalysis_8/PurePursuit_PP_pathFollower_mco_test_8_8_0/"
    planner_path = f"Data/ComparativeAnalysis_8/AgentOff_SAC_Game_mco_train_8_8_0/"
    trajectory_path = f"Data/ComparativeAnalysis_8/AgentOff_SAC_TrajectoryFollower_mco_train_8_8_0/"
    end_path = f"Data/ComparativeAnalysis_8/AgentOff_SAC_endToEnd_mco_train_8_8_0/"

    pp_data = TestLapData(pp_path)
    planner_data = TestLapData(planner_path, 0)
    trajectory_data = TestLapData(trajectory_path, 0)
    end_data = TestLapData(end_path, 0)

    classic_xs = pp_data.generate_state_progress_list()
    planner_xs = planner_data.generate_state_progress_list()
    trajectory_xs = trajectory_data.generate_state_progress_list()
    end_xs = end_data.generate_state_progress_list()
    
    xs = np.linspace(0, 100, 500)
    
    classic_ths = pp_data.states[:, 4]
    classic_ths = np.interp(xs, classic_xs, classic_ths)
    classic_curves = np.diff(classic_ths) / np.diff(xs)
    planner_ths = planner_data.states[:, 4]
    planner_ths = np.interp(xs, planner_xs, planner_ths)
    planner_curves = np.diff(planner_ths) / np.diff(xs)
    trajectory_ths = trajectory_data.states[:, 4]
    trajectory_ths = np.interp(xs, trajectory_xs, trajectory_ths)
    trajectory_curves = np.diff(trajectory_ths) / np.diff(xs)
    end_ths = end_data.states[:, 4]
    end_ths = np.interp(xs, end_xs, end_ths)
    end_curves = np.diff(end_ths) / np.diff(xs)


    # fig, (ax1) = plt.subplots(1, 1, figsize=(6, 1.7), sharex=True)
    plt.figure(1, figsize=(6, 2.19))
    ax1 = plt.gca()
    ax1.plot(xs[:-1], classic_curves, color=pp[0], label="Classic")
    ax1.plot(xs[:-1], planner_curves, color=pp[1], label="Full planner")
    ax1.plot(xs[:-1], trajectory_curves, color=pp[2], label="Trajectory follower")
    ax1.plot(xs[:-1], end_curves, color=pp[3], label="End-to-end")

    ax1.set_ylabel("Curvature (rad/m)")
    ax1.set_xlabel("Track progress (%)")
    ax1.legend(ncol=4, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 0.96))

    plt.grid(True)
    plt.tight_layout()
    plt.xlim(10, 60)
    plt.xlim(30, 60)
    plt.gca().yaxis.set_major_locator(MultipleLocator(1.5))
    y_lim = 3
    plt.ylim(-y_lim, y_lim)

    plt.savefig(f"Data/ComparativeAnalysis_8/_Imgs/CompareCurvature_{map_name.upper()}.svg", bbox_inches='tight')
    plt.savefig(f"Data/ComparativeAnalysis_8/_Imgs/CompareCurvature_{map_name.upper()}.pdf", bbox_inches='tight')

    # plt.show()
    
def speed_profile_deviation():
    map_name = "mco"
    pp_path = f"Data/ComparativeAnalysis_8/PurePursuit_PP_pathFollower_mco_test_8_8_0/"
    planner_path = f"Data/ComparativeAnalysis_8/AgentOff_SAC_Game_mco_train_8_8_0/"
    trajectory_path = f"Data/ComparativeAnalysis_8/AgentOff_SAC_TrajectoryFollower_mco_train_8_8_0/"
    end_path = f"Data/ComparativeAnalysis_8/AgentOff_SAC_endToEnd_mco_train_8_8_0/"

    pp_data = TestLapData(pp_path)
    planner_data = TestLapData(planner_path, 0)
    trajectory_data = TestLapData(trajectory_path, 0)
    end_data = TestLapData(end_path, 0)

    classic_xs = pp_data.generate_state_progress_list()
    planner_xs = planner_data.generate_state_progress_list()
    trajectory_xs = trajectory_data.generate_state_progress_list()
    end_xs = end_data.generate_state_progress_list()

    xs = np.linspace(0, 100, 200)
    pp_vs = np.interp(xs, classic_xs, pp_data.states[:, 3])
    planner_vs = np.interp(xs, planner_xs, planner_data.states[:, 3])
    trajectory_vs = np.interp(xs, trajectory_xs, trajectory_data.states[:, 3])
    end_vs = np.interp(xs, end_xs, end_data.states[:, 3])

    plt.figure(1, figsize=(6, 1.9))
    ax1 = plt.gca()
    plt.plot(xs, planner_vs - pp_vs, label="Full planner", color=pp[1])
    plt.plot(xs, trajectory_vs - pp_vs, color=pp[2], label="Trajectory follower")
    plt.plot(xs, end_vs - pp_vs, color=pp[3], label="End-to-end")

    ax1.set_ylabel("Speed \ndifference m/s", fontsize=10)
    ax1.set_xlabel("Track progress (%)")
    ax1.legend(ncol=4, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 0.96))

    plt.grid(True)
    plt.tight_layout()
    plt.xlim(10, 60)
    plt.gca().yaxis.set_major_locator(MultipleLocator(2.5))

    plt.savefig(f"Data/ComparativeAnalysis_8/_Imgs/SpeedDifference_{map_name.upper()}.svg", bbox_inches='tight')
    plt.savefig(f"Data/ComparativeAnalysis_8/_Imgs/SpeedDifference_{map_name.upper()}.pdf", bbox_inches='tight')

    # plt.show()
    

def lateral_deviation():
    map_name = "mco"
    pp_path = f"Data/ComparativeAnalysis_8/PurePursuit_PP_pathFollower_mco_test_8_8_0/"
    planner_path = f"Data/ComparativeAnalysis_8/AgentOff_SAC_Game_mco_train_8_8_0/"
    trajectory_path = f"Data/ComparativeAnalysis_8/AgentOff_SAC_TrajectoryFollower_mco_train_8_8_0/"
    end_path = f"Data/ComparativeAnalysis_8/AgentOff_SAC_endToEnd_mco_train_8_8_0/"

    pp_data = TestLapData(pp_path)
    planner_data = TestLapData(planner_path, 0)
    trajectory_data = TestLapData(trajectory_path, 0)
    end_data = TestLapData(end_path, 0)

    classic_xs = pp_data.generate_state_progress_list()
    planner_xs = planner_data.generate_state_progress_list()
    trajectory_xs = trajectory_data.generate_state_progress_list()
    end_xs = end_data.generate_state_progress_list()
    
    classic_ds = pp_data.calculate_deviations() * 100
    planner_ds = planner_data.calculate_deviations() * 100
    trajectory_ds = trajectory_data.calculate_deviations() * 100
    end_ds = end_data.calculate_deviations() * 100

    plt.figure(1, figsize=(6, 1.9))
    ax1 = plt.gca()
    ax1.plot(classic_xs, classic_ds, color=pp[0], label="Classic")
    ax1.plot(planner_xs, planner_ds, color=pp[1], label="Full planner")
    ax1.plot(trajectory_xs, trajectory_ds, color=pp[2], label="Trajectory follower")
    ax1.plot(end_xs, end_ds, color=pp[3], label="End-to-end")

    ax1.set_ylabel("Lateral \ndeviation (cm)", fontsize=9)
    ax1.set_xlabel("Track progress (%)")
    ax1.legend(ncol=4, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 0.96))

    plt.grid(True)
    plt.tight_layout()
    plt.xlim(10, 50)
    plt.gca().yaxis.set_major_locator(MultipleLocator(20))

    plt.savefig(f"Data/ComparativeAnalysis_8/_Imgs/LateralDeviation_{map_name.upper()}.svg", bbox_inches='tight')
    plt.savefig(f"Data/ComparativeAnalysis_8/_Imgs/LateralDeviation_{map_name.upper()}.pdf", bbox_inches='tight')

    # plt.show()


# speed_profile_comparison()
# speed_profile_deviation()
# lateral_deviation()
curvature_profile_comparison()
