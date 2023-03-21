import numpy as np
import os, glob

from RacingDRL.Planners.TrackLine import TrackLine

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
    print(f"State: ({np.amin(state_set)}, {np.amax(state_set)}) --> Action: ({np.amin(action_set, axis=0)}, {np.amax(action_set, axis=0)})")
    
def transform_waypoints(wpts, position, orientation):
    new_pts = wpts - position
    new_pts = new_pts @ np.array([[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
    
    return new_pts
    
def generate_game_state(state, scan, track, n_wpts=10):
    idx, dists = track.get_trackline_segment(state[0:2])
        
    upcomings_inds = np.arange(idx, idx+n_wpts)
    if idx + n_wpts >= track.N:
        n_start_pts = idx + n_wpts - track.N
        upcomings_inds[n_wpts - n_start_pts:] = np.arange(0, n_start_pts)
        
    upcoming_pts = track.wpts[upcomings_inds]
    
    relative_pts = transform_waypoints(upcoming_pts, np.array([state[0:2]]), state[4])
    
    speed = state[3] / 8
    anglular_vel = state[5] / 3.14
    steering_angle = state[2] / 0.4
    scaled_state = np.array([speed, anglular_vel, steering_angle])
    
    scaled_scan = np.clip(scan/10, 0, 1)
    
    state = np.concatenate((scaled_scan, relative_pts.flatten(), scaled_state))
    
    return state

    
def build_game_set():
    state_set = [] 
    action_set = []
    
    
    for map_name in ["esp", "aut", "mco", "gbr"]:
        scan_data = np.load(folder + f"RawData/PurePursuit_{map_name}_DataGen_2_Lap_0_scans.npy")
        history_data = np.load(folder + f"RawData/PurePursuit_{map_name}_DataGen_2_Lap_0_history.npy")
        states = history_data[:, :7]
        actions = history_data[:, 7:]

        track = TrackLine(map_name, False)
        prev_scan = np.zeros(20)
        for i, (s, a, scan) in enumerate(zip(states, actions, scan_data)):
            scaled_state = generate_game_state(s, scan, track, n_wpts=10)
            state_set.append(scaled_state)
            scaled_action = a / np.array([0.4, 8])
            action_set.append(scaled_action)
            
    
    state_set = np.array(state_set)
    action_set = np.array(action_set)
    if not os.path.exists(folder + f"DataSets/"): os.mkdir(folder + f"DataSets/")
    np.save(folder + "DataSets/PurePursuit_Game_states.npy", state_set)
    np.save(folder + "DataSets/PurePursuit_Game_actions.npy", action_set)

    print(f"StateShape: {state_set.shape} --> ActionShape: {action_set.shape}")
    print(f"State: ({np.amin(state_set, axis=0)}, {np.amax(state_set, axis=0)}) --> Action: ({np.amin(action_set, axis=0)}, {np.amax(action_set, axis=0)})")
    

# build_endToEnd_set()    
build_game_set()


