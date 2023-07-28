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
# algorithm = "TD3"
algorithm = "SAC"
rep_number = 0
base = f"Data/FinalExperiment_{set_number}/"
names = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]
sub_paths = [f"AgentOff_{algorithm}_Game_{map_name}_{id_name}_8_{set_number}_{rep_number}/",
                f"AgentOff_{algorithm}_TrajectoryFollower_{map_name}_{id_name}_8_{set_number}_{rep_number}/",
                f"AgentOff_{algorithm}_endToEnd_{map_name}_{id_name}_8_{set_number}_{rep_number}/", 
                f"PurePursuit_PP_pathFollower_{map_name}_TAL_8_{set_number}_{rep_number}/"]

lap_n = 0
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
    
    

def speed_profile_comparison():
    
    data_list = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths)]
    xs_list = [d.generate_state_progress_list() for d in data_list]
    xs = np.linspace(0, 100, 200)
    speed_list = [np.interp(xs, xs_list[i], data_list[i].states[:, 3]) for i in range(len(data_list))]
    
    plt.figure(1, figsize=(6, 1.9))
    ax1 = plt.gca()
    for n, name in enumerate(names):
        ax1.plot(xs, speed_list[n], color=color_pallet[n], label=name)

    ax1.set_ylabel("Speed m/s")
    ax1.set_xlabel("Track progress (%)")
    ax1.legend(ncol=4, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 0.96))

    plt.grid(True)
    plt.tight_layout()
    plt.xlim(10, 60)

    plt.savefig(f"{base}Imgs/CompareSpeed_{map_name.upper()}.svg", bbox_inches='tight')
    plt.savefig(f"{base}Imgs/CompareSpeed_{map_name.upper()}.pdf", bbox_inches='tight')

def curvature_profile_comparison():
    data_list = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths)]
    
    xs = np.linspace(0, 100, 200)
    curve_list = [d.calculate_curvature(200) for d in data_list]

    plt.figure(1, figsize=(6, 1.9))
    ax1 = plt.gca()
    for n, name in enumerate(names):
        ax1.plot(xs[:-1], np.abs(curve_list[n]), color=color_pallet[n], label=name)
    
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

    plt.savefig(f"{base}Imgs/CompareCurvature_{map_name.upper()}.svg", bbox_inches='tight')
    plt.savefig(f"{base}Imgs/CompareCurvature_{map_name.upper()}.pdf", bbox_inches='tight')

    # plt.show()
    
def speed_profile_deviation():

    data_list = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths)]
    xs_list = [d.generate_state_progress_list() for d in data_list]
    xs = np.linspace(0, 100, 200)
    speed_list = [np.interp(xs, xs_list[i], data_list[i].states[:, 3]) for i in range(len(data_list))]
    
    plt.figure(1, figsize=(6, 1.9))
    ax1 = plt.gca()
    for n, name in enumerate(names[:-1]):
        ax1.plot(xs, speed_list[n] - speed_list[-1], color=color_pallet[n], label=name)

    ax1.set_ylabel("Speed \n deviation (m/s)", fontsize=10)
    ax1.set_xlabel("Track progress (%)")
    ax1.legend(ncol=4, fontsize=10, loc='lower center', bbox_to_anchor=(0.5, 0.96))

    plt.grid(True)
    plt.tight_layout()
    plt.xlim(10, 60)
    # plt.xlim(-2, 80)
    plt.gca().yaxis.set_major_locator(MultipleLocator(2.5))

    plt.savefig(f"{base}Imgs/SpeedDifference_{map_name.upper()}.svg", bbox_inches='tight')
    plt.savefig(f"{base}Imgs/SpeedDifference_{map_name.upper()}.pdf", bbox_inches='tight')

    # plt.show()
    


def speed_distributions():
    data_list = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths)]
    
    fig, axes = plt.subplots(1, 4, figsize=(6, 1.7), sharex=False, sharey=True)
        
    state_speed_list = [d.states[:, 3] for d in data_list]
        
    bins = np.arange(2, 8, 0.5)
    for i in range(len(data_list)):
        axes[i].hist(state_speed_list[i], color=color_pallet[i], alpha=0.65, bins=bins, density=True)
        
        axes[i].set_xlim(2, 8)
        axes[i].grid(True)
        axes[i].set_title(names[i], fontsize=10)
        axes[i].xaxis.set_tick_params(labelsize=8)
        
    # axes[0].set_ylim(0, 110)
    fig.text(0.54, 0.02, "Vehicle Speed (m/s)", fontsize=10, ha='center')
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].yaxis.set_major_locator(MultipleLocator(0.15))
    axes[0].yaxis.set_tick_params(labelsize=8)
    
    plt.tight_layout()
    name = f"{base}Imgs/SpeedDistributions{map_name.upper()}"
    std_img_saving(name, True)


def slip_distributions():
    data_list = [TestLapData(base + p, lap_list[i]) for i, p in enumerate(sub_paths)]
    
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


# slip_distributions()
