import numpy as np
import os, glob
from matplotlib import pyplot as plt

from FTenthRacingDRL.Planners.TrackLine import TrackLine


set_n = 1
# load_folder = f"Data/PurePursuitDataGen_{set_n}/"
load_folder = f"Data/GenerateDataSet_{set_n}/"


N_BEAMS = 60
    
def calculate_inds(n_beams):
    inds = np.arange(0, 60, round(60/(n_beams-1)))
    if len(inds) < n_beams:
        inds = np.append(inds, 59)
    print(f"inds: {inds} --> {n_beams} --> {len(inds)}")
    
    return inds

def generate_single_endToEnd_state(_state, scan, prev_scans, _track, _n_wpts, n_beams):
    inds = calculate_inds(n_beams)
    scaled_state = scan[inds] / 10
    scaled_state = np.clip(scaled_state, -1, 1)
    
    return scaled_state

def generate_speed_endToEnd_state(state, scan, prev_scans, _track, _n_wpts, n_beams):
    inds = calculate_inds(n_beams)
    scaled_state = scan[inds] / 10
    speed = state[3] / 8
    full_state = np.concatenate((scaled_state, np.array([speed])))
    full_state = np.clip(full_state, -1, 1)
    
    return full_state
    
def generate_double_endToEnd_state(_state, scan, prev_scans, _track, _n_wpts, n_beams):
    inds = calculate_inds(n_beams)
    prev_scan = prev_scans[0]
    scaled_state = np.concatenate((prev_scan[inds], scan[inds])) / 10
    scaled_state = np.clip(scaled_state, -1, 1)
    
    return scaled_state
        
def generate_doubleSpeed_endToEnd_state(state, scan, prev_scans, _track, _n_wpts, n_beams):
    inds = calculate_inds(n_beams)
    prev_scan = prev_scans[0]
    speed = state[3] / 8
    scaled_state = np.concatenate((prev_scan[inds]/ 10, scan[inds]/ 10, np.array([speed]))) 
    scaled_state = np.clip(scaled_state, -1, 1)
    
    return scaled_state    

def generate_tripple_endToEnd_state(_state, scan, prev_scans, _track, _n_wpts, n_beams):
    inds = calculate_inds(n_beams)
    prev_scan = prev_scans[0, :]
    pp_scan = prev_scans[1, :]
    scaled_state = np.concatenate((prev_scan[inds], pp_scan[inds], scan[inds])) / 10
    scaled_state = np.clip(scaled_state, -1, 1)
    
    return scaled_state

    

def build_state_data_set(save_folder, data_tag, state_gen_fcn, n_beams):
    state_set = [] 
    
    for map_name in ["esp", "aut", "mco", "gbr"]:
        scan_data = np.load(load_folder + f"RawData/PurePursuit_{map_name}_DataGen_{set_n}_Lap_0_scans.npy")
        history_data = np.load(load_folder + f"RawData/PurePursuit_{map_name}_DataGen_{set_n}_Lap_0_history.npy")
        states = history_data[:, :7]

        track = TrackLine(map_name, False)
        prev_scans = np.zeros((2, scan_data[0].shape[0]))
        for i in range(len(states)):
            scaled_state = state_gen_fcn(states[i], scan_data[i], prev_scans, track, 10, n_beams)
            state_set.append(scaled_state)
            prev_scans = np.roll(prev_scans, 1, axis=0)
            prev_scans[0] = scan_data[i]
            
    
    state_set = np.array(state_set)
    if not os.path.exists(save_folder + f"DataSets/"): os.mkdir(save_folder + f"DataSets/")
    np.save(save_folder + f"DataSets/PurePursuit_{data_tag}_states.npy", state_set)

    print(f"StateShape: {state_set.shape} ")
    print(f"State Minimum: {np.amin(state_set, axis=0)}")
    print(f"State Maximum: {np.amax(state_set, axis=0)}")
    
    # plt.figure(1)
    # for i in range(20):
    #     plt.plot(state_set[:300, i])
        
    # plt.show()

def build_action_data_set(save_folder):
    normal_action = np.array([0.4, 8])
    
    action_set = []
    for map_name in ["esp", "aut", "mco", "gbr"]:
        history_data = np.load(load_folder + f"RawData/PurePursuit_{map_name}_DataGen_{set_n}_Lap_0_history.npy")
        actions = history_data[:, 7:]

        action_set.append(actions / normal_action)
        
    action_set = np.concatenate(action_set, axis=0)
            
    if not os.path.exists(save_folder + f"DataSets/"): os.mkdir(save_folder + f"DataSets/")
    np.save(save_folder + "DataSets/PurePursuit_actions.npy", action_set)

    print(f"ActionShape: {action_set.shape}")
    print(f"Action: ({np.amin(action_set, axis=0)}, {np.amax(action_set, axis=0)})")
    


def build_endToEnd_nBeams():
    save_folder = f"NetworkFitting/EndToEnd_nBeams_{set_n}/"

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    build_action_data_set(save_folder)

    build_state_data_set(save_folder, "endToEnd_5", generate_double_endToEnd_state, 5)
    build_state_data_set(save_folder, "endToEnd_12", generate_double_endToEnd_state, 12)
    build_state_data_set(save_folder, "endToEnd_10", generate_double_endToEnd_state, 10)
    build_state_data_set(save_folder, "endToEnd_15", generate_double_endToEnd_state, 15)
    build_state_data_set(save_folder, "endToEnd_20", generate_double_endToEnd_state, 20)
    build_state_data_set(save_folder, "endToEnd_30", generate_double_endToEnd_state, 30)
    build_state_data_set(save_folder, "endToEnd_60", generate_double_endToEnd_state, 60)

def build_endToEnd_stacking():
    save_folder = f"NetworkFitting/EndToEnd_stacking_{set_n}/"

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    build_action_data_set(save_folder)

    build_state_data_set(save_folder, "endToEnd_Single", generate_single_endToEnd_state, 20)
    build_state_data_set(save_folder, "endToEnd_Double", generate_double_endToEnd_state, 20)
    build_state_data_set(save_folder, "endToEnd_Triple", generate_tripple_endToEnd_state, 20)
    build_state_data_set(save_folder, "endToEnd_Speed", generate_speed_endToEnd_state, 20)
    build_state_data_set(save_folder, "endToEnd_DoubleSpeed", generate_doubleSpeed_endToEnd_state, 20)
    
    
def build_endToEnd_comparision_set():
    experiment_name = "comparison"
    save_folder = f"NetworkFitting/{experiment_name}_{set_n}/"

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # build_action_data_set(save_folder)

    build_state_data_set(save_folder, "endToEnd", generate_doubleSpeed_endToEnd_state, 20)
    # build_state_data_set(save_folder, "endToEnd_Single", generate_single_endToEnd_state, 20)
    
if __name__ == "__main__":
    # build_endToEnd_nBeams()
    # build_endToEnd_stacking()
    
    build_endToEnd_comparision_set()

