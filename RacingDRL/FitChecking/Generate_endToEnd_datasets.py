import numpy as np
import os, glob
from matplotlib import pyplot as plt

from RacingDRL.Planners.TrackLine import TrackLine


set_n = 3
load_folder = f"Data/PurePursuitDataGen_{set_n}/"
save_folder = f"NetworkFitting/EndToEnd_{set_n}/"

if not os.path.exists(save_folder):
    os.mkdir(save_folder)


N_BEAMS = 60
    
def generate_endToEnd_state(_state, scan, prev_scan, _track, _n_wpts, n_beams):
    # step = round(N_BEAMS/(n_beams-1))
    # print(f"step: {N_BEAMS/(n_beams-1)} -- {step}")
    inds = np.arange(0, 60, round(60/(n_beams-1)))
    if len(inds) < n_beams:
        inds = np.append(inds, 59)
    print(f"inds: {inds} --> {n_beams} --> {len(inds)}")
    scaled_state = np.concatenate((prev_scan[inds], scan[inds])) / 10
    scaled_state = np.clip(scaled_state, -1, 1)
    
    return scaled_state
    
def generate_endToEndHalf_state(_state, scan, prev_scan, _track, _n_wpts):
    scaled_state = np.concatenate((prev_scan[0:12], scan[0:12])) / 10
    scaled_state = np.clip(scaled_state, -1, 1)
    
    return scaled_state

def generate_endToEndSingle_state(_state, scan, prev_scan, _track, _n_wpts):
    scaled_state = scan / 10
    scaled_state = np.clip(scaled_state[0:1], -1, 1)
    
    return scaled_state
    

def build_state_data_set(data_tag, state_gen_fcn, n_beams):
    state_set = [] 
    
    for map_name in ["esp", "aut", "mco", "gbr"]:
        scan_data = np.load(load_folder + f"RawData/PurePursuit_{map_name}_DataGen_{set_n}_Lap_0_scans.npy")
        history_data = np.load(load_folder + f"RawData/PurePursuit_{map_name}_DataGen_{set_n}_Lap_0_history.npy")
        states = history_data[:, :7]

        track = TrackLine(map_name, False)
        prev_scan = np.zeros_like(scan_data[0])
        for i in range(len(states)):
            scaled_state = state_gen_fcn(states[i], scan_data[i], prev_scan, track, 10, n_beams)
            state_set.append(scaled_state)
            prev_scan = scan_data[i]
            
    
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

def build_action_data_set():
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
    


build_action_data_set()

build_state_data_set("endToEnd_5", generate_endToEnd_state, 5)
build_state_data_set("endToEnd_10", generate_endToEnd_state, 10)
build_state_data_set("endToEnd_12", generate_endToEnd_state, 12)
build_state_data_set("endToEnd_15", generate_endToEnd_state, 15)
build_state_data_set("endToEnd_20", generate_endToEnd_state, 20)
build_state_data_set("endToEnd_30", generate_endToEnd_state, 30)
build_state_data_set("endToEnd_60", generate_endToEnd_state, 60)


