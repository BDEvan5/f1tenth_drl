import numpy as np
from RacingDRL.Planners.StdTrack import StdTrack
from matplotlib import pyplot as plt

def select_architecture(run, conf):
    if run.state_vector == "end_to_end":
        architecture = ArchEndToEnd(run, conf)
    elif run.state_vector == "path_follower":
        architecture = ArchPathFollower(run, conf)
    else:
        raise ValueError("Unknown state vector type: " + run.state_vector)
            
    return architecture

class ArchEndToEnd:
    def __init__(self, run, conf):
        self.state_space = conf.n_beams 
        self.range_finder_scale = conf.range_finder_scale
        self.n_beams = conf.n_beams
        self.max_speed = run.max_speed
        self.max_steer = conf.max_steer

        self.action_space = 2

        self.n_scans = run.n_scans
        self.scan_buffer = np.zeros((self.n_scans, self.n_beams))
        self.state_space *= self.n_scans

    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """
            
        scan = np.array(obs['scans'][0]) 

        scaled_scan = scan/self.range_finder_scale
        scan = np.clip(scaled_scan, 0, 1)

        if self.scan_buffer.all() ==0: # first reading
            for i in range(self.n_scans):
                self.scan_buffer[i, :] = scan 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        nn_obs = np.reshape(self.scan_buffer, (self.n_beams * self.n_scans))

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        speed = (nn_action[1] + 1) * (self.max_speed  / 2 - 0.5) + 1
        speed = min(speed, self.max_speed) # cap the speed

        action = np.array([steering_angle, speed])

        return action


class ArchPathFollower:
    def __init__(self, run, conf):
        self.max_speed = run.max_speed
        self.max_steer = conf.max_steer
        self.n_wpts = 10
        self.state_space = self.n_wpts * 2 + 4

        self.action_space = 2

        # self.n_stacked_states = run.n_stacked_states

        self.track = StdTrack(run.map_name)
        self.previous_action = np.zeros(2)
    
    def transform_obs(self, obs):
        idx, dists = self.track.get_trackline_segment([obs['poses_x'][0], obs['poses_y'][0]])
        
        upcomings_inds = np.arange(idx, idx+self.n_wpts)
        if idx + self.n_wpts >= self.track.N:
            n_start_pts = idx + self.n_wpts - self.track.N
            upcomings_inds[self.n_wpts - n_start_pts:] = np.arange(0, n_start_pts)
            
        upcoming_pts = self.track.wpts[upcomings_inds]
        
        relative_pts = transform_waypoints(upcoming_pts, np.array([obs['poses_x'][0], obs['poses_y'][0]]), obs['poses_theta'][0])
        
        speed = obs['linear_vels_x'][0]
        anglular_vel = obs['ang_vels_z'][0]
        
        # print(relative_pts)
        # print(f"Speed: {speed}, anglular_vel: {anglular_vel}, prev action: {self.previous_action}")
        
        state = np.concatenate((relative_pts.flatten(), np.array([speed, anglular_vel, self.previous_action[0], self.previous_action[1]])))
        
        # plot_state(state)
        
        return state
    
    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        speed = (nn_action[1] + 1) * (self.max_speed  / 2 - 0.5) + 1
        speed = min(speed, self.max_speed) # cap the speed

        action = np.array([steering_angle, speed])
        self.previous_action = action

        return action
        
def transform_waypoints(wpts, position, orientation):
    new_pts = wpts - position
    new_pts = new_pts @ np.array([[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
    
    return new_pts
    
def plot_state(state):
    pts = np.reshape(state[:20], (10, 2))
    
    plt.figure(1)
    plt.clf()
    plt.plot(pts[:, 0], pts[:, 1], 'ro-')
    
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    plt.pause(0.00001)
    # plt.show()
    
    
    
    
class ArchHybrid:
    def __init__(self):
        pass
    
    def transform_obs(self, obs):
        raise NotImplementedError
    
    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        speed = (nn_action[1] + 1) * (self.max_speed  / 2 - 0.5) + 1
        speed = min(speed, self.max_speed) # cap the speed

        action = np.array([steering_angle, speed])
        self.previous_action = action

        return action