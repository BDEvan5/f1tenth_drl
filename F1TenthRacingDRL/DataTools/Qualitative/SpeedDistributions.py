import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
from F1TenthRacingDRL.DataTools.MapData import MapData
from F1TenthRacingDRL.Planners.TrackLine import TrackLine 

from F1TenthRacingDRL.DataTools.plotting_utils import *

set_number = 1
id_name = "TAL"
# id_name = "Cth"
map_name = "gbr"
# map_name = "mco"
rep_number = 0
pp_n = 0
base = f"Data/FinalExperiment_{set_number}/"
names = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]
algorithm = "TD3"
sub_paths_td3 = [f"AgentOff_{algorithm}_Game_{map_name}_{id_name}_8_{set_number}_{rep_number}/",
                f"AgentOff_{algorithm}_TrajectoryFollower_{map_name}_{id_name}_8_{set_number}_{rep_number}/",
                f"AgentOff_{algorithm}_endToEnd_{map_name}_{id_name}_8_{set_number}_{rep_number}/", 
                f"PurePursuit_PP_pathFollower_{map_name}_TAL_8_{set_number}_{pp_n}/"]


algorithm = "SAC"
sub_paths_sac = [f"AgentOff_{algorithm}_Game_{map_name}_{id_name}_8_{set_number}_{rep_number}/",
                f"AgentOff_{algorithm}_TrajectoryFollower_{map_name}_{id_name}_8_{set_number}_{rep_number}/",
                f"AgentOff_{algorithm}_endToEnd_{map_name}_{id_name}_8_{set_number}_{rep_number}/", 
                f"PurePursuit_PP_pathFollower_{map_name}_TAL_8_{set_number}_{pp_n}/"]

lap_n = 2
lap_list = np.ones(4, dtype=int) * lap_n


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
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}.npy")
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
    
    


def speed_distributions_algs():
    data_list_t = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths_td3)]
    data_list_s = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths_sac)]
    
    fig, axes = plt.subplots(2, 4, figsize=(6, 2.5), sharex=True, sharey=True)
        
    state_speed_list_t = [d.states[:, 3] for d in data_list_t]
    state_speed_list_s = [d.states[:, 3] for d in data_list_s]
        
    bins = np.arange(2, 8, 0.5)
    for i in range(len(data_list_t)):
        axes[0, i].hist(state_speed_list_t[i], color=color_pallet[i], alpha=0.65, bins=bins, density=True)
        
        axes[0, i].set_xlim(2, 8)
        axes[0, i].grid(True)
        axes[0, i].set_title(names[i], fontsize=10)
        axes[0, i].xaxis.set_tick_params(labelsize=8)

        axes[1, i].hist(state_speed_list_s[i], color=color_pallet[i], alpha=0.65, bins=bins, density=True)
        
        axes[1, i].grid(True)
        axes[1, i].xaxis.set_tick_params(labelsize=8)

        if i < 3:
            y_val = 0.45
            x_val = 5.8
            axes[0, i].text(x_val, y_val, "TD3", fontdict={'fontsize': 10, 'fontweight':'bold'}, bbox=dict(facecolor='white', alpha=0.99, boxstyle='round,pad=0.15', edgecolor='gainsboro'))
            axes[1, i].text(x_val, y_val, "SAC", fontdict={'fontsize': 10, 'fontweight':'bold'}, bbox=dict(facecolor='white', alpha=0.99, boxstyle='round,pad=0.15', edgecolor='gainsboro'))

        
    # axes[0].set_ylim(0, 110)
    fig.text(0.54, 0.02, "Vehicle Speed (m/s)", fontsize=10, ha='center')
    axes[0, 0].set_ylabel("Density", fontsize=10)
    axes[1, 0].set_ylabel("Density", fontsize=10)
    axes[0, 0].yaxis.set_major_locator(MultipleLocator(0.15))
    axes[0, 0].yaxis.set_tick_params(labelsize=8)
    axes[1, 0].yaxis.set_tick_params(labelsize=8)
    
    plt.tight_layout()
    name = f"{base}Imgs/SpeedDistributions_Algs_{map_name.upper()}"
    std_img_saving(name, True)


def slip_distributions():
    data_list = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths_td3)]
    
    fig, axes = plt.subplots(1, 4, figsize=(6, 1.7), sharex=True, sharey=True)
        
    state_speed_list = [np.abs(d.states[:, 6]) for d in data_list]
        
    bins = np.arange(0, 0.5, 0.05)
    # bins = np.arange(2, 8, 0.5)
    for i in range(len(data_list)):
        axes[i].hist(state_speed_list[i], color=color_pallet[i], bins=bins, alpha=0.65, density=True)
        
        axes[i].set_xlim(-0.05, 0.5)
        axes[i].grid(True)
        axes[i].set_title(names[i], fontsize=10)
        axes[i].xaxis.set_tick_params(labelsize=8)
        
    axes[0].set_ylim(0, 10)
    fig.text(0.54, 0.02, "Vehicle Speed (m/s)", fontsize=10, ha='center')
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    # axes[0].yaxis.set_major_locator(MultipleLocator(0.15))
    axes[0].yaxis.set_tick_params(labelsize=8)
    
    plt.tight_layout()
    name = f"{base}Imgs/SlipDistributions{map_name.upper()}"
    std_img_saving(name, True)



# speed_profile_comparison()
# plt.clf()
# speed_profile_deviation()
# plt.clf()
# speed_distributions()
speed_distributions_algs()


# slip_distributions()
