import numpy as np 
import matplotlib.pyplot as plt
import csv
import trajectory_planning_helpers as tph
from matplotlib.collections import LineCollection
np.printoptions(precision=3, suppress=True)

MAX_KAPPA = 0.8
VEHICLE_WIDTH = 1
RACELINE_STEP = 0.2
MU = 0.7
V_MAX = 8
VEHICLE_MASS = 3.4
ax_max_machine = np.array([[0, 8.5],[8, 8.5]])
ggv = np.array([[0, 8.5, 8.5], [8, 8.5, 8.5]])


class OptimiseMap:
    def __init__(self, map_name, smoothing) -> None:
        self.map_name = map_name

        self.track = None
        self.nvecs = None

        self.vs = None
        self.s_raceline = None
        self.raceline = None
        self.psi_r = None
        self.kappa_ = None

        self.load_track_pts()
        self.make_nvecs(smoothing)
        self.generate_minimum_curvature_path()
        self.plot_minimum_curvature_path()
        self.generate_velocity_profile()
        self.plot_raceline_trajectory()

    def load_track_pts(self):
        track = []
        filename = 'maps/' + self.map_name + "_centerline.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename} in env_map")

        self.N = len(track)
        self.track = track
        
    def make_nvecs(self, smoothing):
        old_track = self.track
        el_lengths = np.linalg.norm(np.diff(self.track[:, :2], axis=0), axis=1)
        psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(self.track[:, :2], el_lengths, False)
        self.nvecs = tph.calc_normal_vectors.calc_normal_vectors(psi)

        crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs)
        print("Crossing: ", crossing)
        # if crossing:
        #     print("Normals are crossing, fixing...")
        #     self.track = tph.spline_approximation.spline_approximation(self.track, 4, smoothing, 0.1, 0.3, True)    

        #     el_lengths = np.linalg.norm(np.diff(self.track[:, 0:2], axis=0), axis=1)
        #     psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(self.track[:, 0:2], el_lengths, False)
        #     self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(psi+np.pi/2)

        #     still_crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs)
        #     print(f"Still crossing {still_crossing}")

        #     plt.figure(5)
        #     plt.clf()
        #     plt.title("Track After Smoothing")
        #     plt.plot(old_track[:, 0], old_track[:, 1], '-', linewidth=2, color='blue')

        #     plt.plot(self.track[:, 0], self.track[:, 1], '-', linewidth=2, color='red')
        #     l1 = self.track[:, 0:2] + self.nvecs * self.track[:, 2][:, None]
        #     l2 = self.track[:, 0:2] - self.nvecs * self.track[:, 3][:, None]

        #     for i in range(len(self.track)):
        #         plt.plot([l1[i, 0], l2[i, 0]], [l1[i, 1], l2[i, 1]], linewidth=1, color='green')

        #     plt.plot(l1[:, 0], l1[:, 1], linewidth=1, color='green')
        #     plt.plot(l2[:, 0], l2[:, 1], linewidth=1, color='green')

        #     print(f"Minimum width --> L: {np.min(self.track[:, 2])}, R: {np.min(self.track[:, 3])}")

        #     if still_crossing: plt.show()
        # plt.show()

    def generate_minimum_curvature_path(self):
        el_lengths = np.linalg.norm(np.diff(self.track[:, 0:2], axis=0), axis=1)
        psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(self.track[:, 0:2], el_lengths, False)
        coeffs_x, coeffs_y, A, normvec_normalized = tph.calc_splines.calc_splines(self.track[:, 0:2], el_lengths, psi_s=psi[0], psi_e=psi[-1])
        self.nvecs = tph.calc_normal_vectors.calc_normal_vectors(psi)
        # psi = psi - np.pi/2
        print(f"Psi -> Start: {psi[0]}, End: {psi[-1]}")

        # alpha, error = tph.opt_min_curv.opt_min_curv(self.track, self.nvecs, A, MAX_KAPPA, VEHICLE_WIDTH, print_debug=True, closed=False, fix_s=True, psi_s=psi[0], psi_e=psi[-1], fix_e=True)
        alpha, error = tph.opt_min_curv.opt_min_curv(self.track, self.nvecs, A, MAX_KAPPA, VEHICLE_WIDTH, print_debug=True, closed=False, fix_s=True, psi_s=psi[0], psi_e=psi[-1], fix_e=True)
        if np.isnan(alpha[0]):
            raise ValueError("Alpha is NaN")

        print(f"Raceline shape: {self.track.shape}")
        print(f"Nvecs shape: {self.nvecs.shape}")
        print(f"Alpha shape: {alpha.shape}")

        self.raceline, A_raceline, coeffs_x_raceline, coeffs_y_raceline, spline_inds_raceline_interp, t_values_raceline_interp, self.s_raceline, spline_lengths_raceline, el_lengths_raceline_interp_cl = tph.create_raceline.create_raceline(self.track[:, 0:2], self.nvecs, alpha, RACELINE_STEP) 
        self.psi_r, self.kappa_r = tph.calc_head_curv_num.calc_head_curv_num(self.raceline, el_lengths_raceline_interp_cl, True)

    def plot_minimum_curvature_path(self):
        plt.figure(3)
        plt.clf()
        plt.title("Minimum Curvature Raceline")
        plt.plot(self.track[:, 0], self.track[:, 1], '-', linewidth=2, color='blue', label="Track")
        plt.plot(self.raceline[:, 0], self.raceline[:, 1], '-', linewidth=2, color='red', label="Raceline")

        l_line = self.track[:, 0:2] + self.nvecs * self.track[:, 2][:, None]
        r_line = self.track[:, 0:2] - self.nvecs * self.track[:, 3][:, None]
        plt.plot(l_line[:, 0], l_line[:, 1], linewidth=1, color='green', label="Boundaries")
        plt.plot(r_line[:, 0], r_line[:, 1], linewidth=1, color='green')
        
        plt.legend()

        plt.tight_layout()
        plt.pause(0.001)

    def generate_velocity_profile(self):
        mu = MU * np.ones(len(self.kappa_r))
        el_lengths = np.linalg.norm(np.diff(self.raceline, axis=0), axis=1)

        self.vs = tph.calc_vel_profile.calc_vel_profile(ax_max_machine, self.kappa_r, el_lengths, False, 0, VEHICLE_MASS, ggv=ggv, mu=mu, v_max=V_MAX, v_start=V_MAX)

        ts = tph.calc_t_profile.calc_t_profile(self.vs, el_lengths, 0)
        print(f"Planned Lap Time: {ts[-1]}")

        acc = tph.calc_ax_profile.calc_ax_profile(self.vs, el_lengths, True)

        save_arr = np.concatenate([self.s_raceline[:, None], self.raceline, self.psi_r[:, None], self.kappa_r[:, None], self.vs[:, None], acc[:, None]], axis=1)

        np.savetxt("maps/"+ self.map_name+ '_raceline.csv', save_arr, delimiter=',')

    def plot_raceline_trajectory(self):
        plt.figure(1)
        plt.clf()
        plt.title("Racing Line Velocity Profile")

        vs = self.vs
        points = self.raceline.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(2, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(3)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line)
        plt.xlim(self.raceline[:, 0].min()-1, self.raceline[:, 0].max()+1)
        plt.ylim(self.raceline[:, 1].min()-1, self.raceline[:, 1].max()+1)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        # plt.show()
        plt.pause(0.001)

def optimise_all_maps():
    map_list = ['aut', 'esp', 'gbr', 'mco']
    smoothing = [30, 30, 40, 60, 250]
    for i, map_name in enumerate(map_list):
        opti = OptimiseMap(map_name, smoothing[i])

  
def run_profiling(function, name):
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    function()
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    with open(f"Data/profile_{name}.txt", "w") as f:
        ps.print_stats()
        f.write(s.getvalue())

def profile_aut():
    OptimiseMap('aut', 30)

if __name__ == "__main__":
    # OptimiseMap('autW', 250)
    # optimise_all_maps()
    # opti = OptimiseMap("aut", 0)
    # opti = OptimiseMap("mco", 0)

    # plt.show()
    # run_profiling(profile_aut, 'GenerateAUT')
    OptimiseMap('aut', 0)
    # plt.show()
