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

    # plt.show()


speed_profile_comparison()

