import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
from F1TenthRacingDRL.DataTools.MapData import MapData
from F1TenthRacingDRL.Planners.TrackLine import TrackLine 

from F1TenthRacingDRL.DataTools.plotting_utils import *

set_number = 1
map_name = "gbr"
algorithm = "SAC"
rep_number = 0
base = f"Data/FinalExperiment_{set_number}/"
names = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]


lap_n = 4
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
    
    



def slip_distributions():
    fig, axes = plt.subplots(2, 4, figsize=(6, 2.8), sharex=True, sharey=True)
        
    for id, id_name in enumerate(["Cth", "TAL"]):
        sub_paths = [f"AgentOff_{algorithm}_Game_{map_name}_{id_name}_8_{set_number}_{rep_number}/",
                        f"AgentOff_{algorithm}_TrajectoryFollower_{map_name}_{id_name}_8_{set_number}_{rep_number}/",
                        f"AgentOff_{algorithm}_endToEnd_{map_name}_{id_name}_8_{set_number}_{rep_number}/", 
                        f"PurePursuit_PP_pathFollower_{map_name}_TAL_8_{set_number}_{rep_number}/"]

        data_list = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths)]
        state_speed_list = [np.abs(d.states[:, 6])  for d in data_list]
        # state_speed_list = [np.abs(d.states[:, 6]) * 180/np.pi for d in data_list]
            
        # bins = np.arange(0, 36, 4)
        bins = np.arange(0, 0.5, 0.05)
        for i in range(len(data_list)):
            # axes[id, i].hist(state_speed_list[i], color=color_pallet[i], alpha=0.65, density=True)
            axes[id, i].hist(state_speed_list[i], color=color_pallet[i], bins=bins, alpha=0.65, density=True)
            
            axes[id, i].grid(True)
            axes[0, i].set_title(names[i], fontsize=10)

            if i != 3:
                axes[id, i].text(0.3, 6, id_name.upper(), fontdict={'fontsize': 10, 'fontweight':'bold'}, bbox=dict(facecolor='white', alpha=0.99, boxstyle='round,pad=0.15', edgecolor='gainsboro'))

    axes[0, 0].set_xlim(-0.05, 0.5)
    # axes[0, 0].set_xlim(-0.05, 36)
    axes[0, 0].xaxis.set_tick_params(labelsize=8)
        
    # axes[0, 0].set_ylim(0, 20)
    axes[0, 0].set_yscale("log")
    fig.text(0.56, 0.0, "Slip angle (rad)", fontsize=10, ha='center')
    axes[0, 0].set_ylabel("Density", fontsize=10)
    axes[1, 0].set_ylabel("Density", fontsize=10)
    # axes[0].yaxis.set_major_locator(MultipleLocator(0.15))
    axes[0, 0].yaxis.set_tick_params(labelsize=8)
    axes[1, 0].yaxis.set_tick_params(labelsize=8)
    
    plt.tight_layout()
    name = f"{base}Imgs/SlipDistributions{map_name.upper()}_Lap{lap_n}"
    std_img_saving(name, True)



# speed_profile_comparison()
# plt.clf()
# speed_profile_deviation()
# plt.clf()
# speed_distributions()


slip_distributions()
