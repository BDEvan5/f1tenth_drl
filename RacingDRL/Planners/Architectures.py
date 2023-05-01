import numpy as np
from RacingDRL.Planners.TrackLine import TrackLine
from matplotlib import pyplot as plt

def select_architecture(run, conf):
    if run.state_vector == "endToEnd":
        architecture = EndArchitecture(run, conf)
    elif run.state_vector == "TrajectoryFollower":
        architecture = TrajectoryArchitecture(run, conf)
    elif run.state_vector == "Game":
        architecture = PlanningArchitecture(run, conf)
    else:
        raise ValueError("Unknown state vector type: " + run.state_vector)
            
    return architecture

class EndArchitecture:
    def __init__(self, run, conf):
        self.range_finder_scale = 10
        self.n_beams = run.num_beams
        self.max_speed = run.max_speed
        self.max_steer = conf.max_steer

        self.action_space = 2

        self.state_space = self.n_beams + 1 

    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """
        scan = np.array(obs['scans'][0]) 
        speed = obs['linear_vels_x'][0]/self.max_speed
        scan = np.clip(scan/self.range_finder_scale, 0, 1)
        nn_obs = np.concatenate((scan, [speed]))

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        speed = (nn_action[1] + 1) * (self.max_speed  / 2 - 0.5) + 1
        speed = min(speed, self.max_speed) # cap the speed

        action = np.array([steering_angle, speed])

        return action


class PlanningArchitecture:
    def __init__(self, run, conf):
        self.max_speed = run.max_speed
        self.max_steer = conf.max_steer
        self.n_wpts = 10 
        self.n_beams = 20
        self.waypoint_scale = 2.5
        self.state_space = self.n_wpts * 2 + 3 + self.n_beams
        self.action_space = 2

        self.track = TrackLine(run.map_name, False)
    
    def transform_obs(self, obs):
        idx, dists = self.track.get_trackline_segment([obs['poses_x'][0], obs['poses_y'][0]])
        
        upcomings_inds = np.arange(idx, idx+self.n_wpts)
        if idx + self.n_wpts >= self.track.N:
            n_start_pts = idx + self.n_wpts - self.track.N
            upcomings_inds[self.n_wpts - n_start_pts:] = np.arange(0, n_start_pts)
            
        upcoming_pts = self.track.wpts[upcomings_inds]
        
        relative_pts = transform_waypoints(upcoming_pts, np.array([obs['poses_x'][0], obs['poses_y'][0]]), obs['poses_theta'][0])
        relative_pts /= self.waypoint_scale
        relative_pts = relative_pts.flatten()
        
        speed = obs['linear_vels_x'][0] / self.max_speed
        anglular_vel = obs['ang_vels_z'][0] / np.pi
        steering_angle = obs['steering_deltas'][0] / self.max_steer
        
        scan = np.array(obs['scans'][0]) 
        scan = np.clip(scan/10, 0, 1)
        
        motion_variables = np.array([speed, anglular_vel, steering_angle])
        state = np.concatenate((scan, relative_pts.flatten(), motion_variables))
        
        return state
    
    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        speed = (nn_action[1] + 1) * (self.max_speed  / 2 - 0.5) + 1
        speed = min(speed, self.max_speed) # cap the speed

        action = np.array([steering_angle, speed])
        self.previous_action = action

        return action


class TrajectoryArchitecture:
    def __init__(self, run, conf):
        self.max_speed = run.max_speed
        self.max_steer = conf.max_steer
        self.n_wpts = run.n_waypoints
        self.state_space = self.n_wpts * 3 + 3
        self.waypoint_scale = 2.5

        self.action_space = 2
        self.track = TrackLine(run.map_name, True)
    
    def transform_obs(self, obs):
        pose = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        idx, dists = self.track.get_trackline_segment(pose)
        
        speed = obs['linear_vels_x'][0] / self.max_speed
        anglular_vel = obs['ang_vels_z'][0] / np.pi
        steering_angle = obs['steering_deltas'][0] / self.max_steer
        
        upcomings_inds = np.arange(idx+1, idx+self.n_wpts+1)
        if idx + self.n_wpts + 1 >= self.track.N:
            n_start_pts = idx + self.n_wpts + 1 - self.track.N
            upcomings_inds[self.n_wpts - n_start_pts:] = np.arange(0, n_start_pts)
            
        upcoming_pts = self.track.wpts[upcomings_inds]
        
        relative_pts = transform_waypoints(upcoming_pts, np.array([obs['poses_x'][0], obs['poses_y'][0]]), obs['poses_theta'][0])
        relative_pts /= self.waypoint_scale
        
        speeds = self.track.vs[upcomings_inds]
        scaled_speeds = speeds / self.max_speed
        # scaled_speeds = (speeds - 1) / (self.max_speed - 1) * 2 - 1
        relative_pts = np.concatenate((relative_pts, scaled_speeds[:, None]), axis=-1)
        
        # plt.figure()
        # plt.plot(relative_pts[:, 0], relative_pts[:, 1], 'r.')
        # plt.plot(upcoming_pts[:, 0], upcoming_pts[:, 1], 'r.')
        # plt.plot(pose[0], pose[1], 'b.')
        
        # plt.show()
        
        relative_pts = relative_pts.flatten()
        motion_variables = np.array([speed, anglular_vel, steering_angle])
        state = np.concatenate((relative_pts, motion_variables))
        
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
    
    
    
