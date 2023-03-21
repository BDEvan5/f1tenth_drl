import numpy as np
import os, glob

folder = "Data/PurePursuitDataGen_2/"

def build_endToEnd_set():
    state_set = [] 
    action_set = []
    
    for map_name in ["esp", "aut", "mco", "gbr"]:
        scan_data = np.load(folder + f"RawData/PurePursuit_{map_name}_DataGen_2_Lap_0_scans.npy")
        history_data = np.load(folder + f"RawData/PurePursuit_{map_name}_DataGen_2_Lap_0_history.npy")
        states = history_data[:, :7]
        actions = history_data[:, 7:]

        prev_scan = np.zeros(20)
        for i, (s, a, scan) in enumerate(zip(states, actions, scan_data)):
            scaled_state = np.concatenate((prev_scan, scan)) / 10
            scaled_state = np.clip(scaled_state, -1, 1)
            state_set.append(scaled_state)
            scaled_action = a / np.array([0.4, 5])
            action_set.append(scaled_action)
            prev_scan = scan
            
    
    state_set = np.array(state_set)
    action_set = np.array(action_set)
    if not os.path.exists(folder + f"DataSets/"): os.mkdir(folder + f"DataSets/")
    np.save(folder + "DataSets/PurePursuit_endToEnd_states.npy", state_set)
    np.save(folder + "DataSets/PurePursuit_endToEnd_actions.npy", action_set)

    print(f"StateShape: {state_set.shape} --> ActionShape: {action_set.shape}")
    print(f"State: ({np.amin(state_set)}, {np.amax(state_set)}) --> Action: ({np.amax(action_set)}, {np.amax(action_set)})")
    

build_endToEnd_set()    

