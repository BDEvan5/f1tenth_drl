import numpy as np
import os, glob
from matplotlib import pyplot as plt

from RacingDRL.Planners.TrackLine import TrackLine

folder = "Data/PurePursuitDataGen_2/"


    
def generate_endToEnd_state(_state, scan, prev_scan, _track, _n_wpts):
    scaled_state = np.concatenate((prev_scan, scan)) / 10
    scaled_state = np.clip(scaled_state, -1, 1)
    
    return scaled_state
    
def transform_waypoints(wpts, position, orientation):
    new_pts = wpts - position
    new_pts = new_pts @ np.array([[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
    
    return new_pts
    
def generate_game_state(state, scan, _prev_scan, track, n_wpts):
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

def generate_trajFollow_state(state, _scan, _prev_scan, track, n_wpts):
    idx, dists = track.get_trackline_segment(state[0:2])
        
    upcomings_inds = np.arange(idx, idx+n_wpts)
    if idx + n_wpts >= track.N:
        n_start_pts = idx + n_wpts - track.N
        upcomings_inds[n_wpts - n_start_pts:] = np.arange(0, n_start_pts)
        
    upcoming_pts = track.wpts[upcomings_inds]
    
    relative_pts = transform_waypoints(upcoming_pts, np.array([state[0:2]]), state[4])
    relative_pts = relative_pts / 2.5
    relative_pts = np.clip(relative_pts, 0, 1) #? ensures correct range
    
    speed = state[3] / 8
    anglular_vel = state[5] / 3.14
    steering_angle = state[2] / 0.4
    scaled_state = np.array([speed, anglular_vel, steering_angle])
    scaled_state = np.clip(scaled_state, -1, 1)
    
    state = np.concatenate((relative_pts.flatten(), scaled_state))
    
    return state

    


def build_state_data_set(data_tag, state_gen_fcn):
    state_set = [] 
    
    for map_name in ["esp", "aut", "mco", "gbr"]:
        scan_data = np.load(folder + f"RawData/PurePursuit_{map_name}_DataGen_2_Lap_0_scans.npy")
        history_data = np.load(folder + f"RawData/PurePursuit_{map_name}_DataGen_2_Lap_0_history.npy")
        states = history_data[:, :7]

        track = TrackLine(map_name, False)
        prev_scan = np.zeros(20)
        for i in range(len(states)):
            scaled_state = state_gen_fcn(states[i], scan_data[i], prev_scan, track, 10)
            state_set.append(scaled_state)
            prev_scan = scan_data[i]
            
    
    state_set = np.array(state_set)
    if not os.path.exists(folder + f"DataSets/"): os.mkdir(folder + f"DataSets/")
    np.save(folder + f"DataSets/PurePursuit_{data_tag}_states.npy", state_set)

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
        history_data = np.load(folder + f"RawData/PurePursuit_{map_name}_DataGen_2_Lap_0_history.npy")
        actions = history_data[:, 7:]

        action_set.append(actions / normal_action)
        
    action_set = np.concatenate(action_set, axis=0)
            
    if not os.path.exists(folder + f"DataSets/"): os.mkdir(folder + f"DataSets/")
    np.save(folder + "DataSets/PurePursuit_actions.npy", action_set)

    print(f"ActionShape: {action_set.shape}")
    print(f"Action: ({np.amin(action_set, axis=0)}, {np.amax(action_set, axis=0)})")
    


# build_action_data_set()
# build_state_data_set("Game", generate_game_state)
# build_state_data_set("endToEnd", generate_endToEnd_state)
build_state_data_set("trajFollow", generate_trajFollow_state)


