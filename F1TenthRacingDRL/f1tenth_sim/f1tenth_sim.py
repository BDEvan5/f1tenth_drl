

from F1TenthRacingDRL.f1tenth_sim.dynamics_simulator import DynamicsSimulator
from F1TenthRacingDRL.f1tenth_sim.laser_models import ScanSimulator2D

import numpy as np
import os
import time
from scipy import interpolate, optimize, spatial

'''
    params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
    mu: surface friction coefficient
    C_Sf: Cornering stiffness coefficient, front
    C_Sr: Cornering stiffness coefficient, rear
    lf: Distance from center of gravity to front axle
    lr: Distance from center of gravity to rear axle
    h: Height of center of gravity
    m: Total mass of the vehicle
    I: Moment of inertial of the entire vehicle about the z axis
    s_min: Minimum steering angle constraint
    s_max: Maximum steering angle constraint
    sv_min: Minimum steering velocity constraint
    sv_max: Maximum steering velocity constraint
    v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
    a_max: Maximum longitudinal acceleration
    v_min: Minimum longitudinal velocity
    v_max: Maximum longitudinal velocity
    width: width of the vehicle in meters
    length: length of the vehicle in meters
'''

class F1TenthSim:
    """
            seed (int, default=12345): seed for random state and reproducibility
            map (str, default='vegas'): name of the map used for the environment. 
            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'
    """
    def __init__(self, map_name, seed=12345):
        self.seed = seed

        self.map_name = "maps/" + map_name
        self.map_path = self.map_name + ".yaml"

        self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

        self.current_time = 0.0

        num_beams = 20
        fov = 4.7
        self.timestep = 0.01

        self.scan_simulator = ScanSimulator2D(num_beams, fov)
        self.scan_simulator.set_map(self.map_path)
        self.dynamics_simulator = DynamicsSimulator(self.params, self.seed, self.timestep)
        self.scan_rng = np.random.default_rng(seed=self.seed)

        self.center_line = np.loadtxt(self.map_name + "_centerline.csv", delimiter=',')[:, :2]
        el_lengths = np.linalg.norm(np.diff(self.center_line, axis=0), axis=1)
        self.s_track = np.insert(np.cumsum(el_lengths), 0, 0)
        self.tck = interpolate.splprep([self.center_line[:, 0], self.center_line[:, 1]], k=3, s=0)[0]

        self.lap_number = -1

    def step(self, action):
        vehicle_state = self.dynamics_simulator.update_pose(action[0], action[1])
        pose = np.append(vehicle_state[0:2], vehicle_state[4])

        scan = self.scan_simulator.scan(np.append(vehicle_state[0:2], vehicle_state[4]), self.scan_rng)

        self.collision = self.check_vehicle_collision(pose)
        self.lap_complete, progress = self.check_lap_complete(pose)
        self.current_time = self.current_time + self.timestep

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        observation = {"scan": scan,
                        "vehicle_state": self.dynamics_simulator.state,
                        "collision": self.collision,
                        "lap_complete": self.lap_complete,
                        "laptime": self.current_time,
                        "progress": progress}
        
        done = self.collision or self.lap_complete

        if self.collision:
            print(f"{self.lap_number} COLLISION: Time: {self.current_time:.2f}, Progress: {100*progress:.1f}")
        elif self.lap_complete:
            print(f"{self.lap_number} LAP COMPLETE: Time: {self.current_time:.2f}, Progress: {(100*progress):.1f}")

        return observation, done

    def check_lap_complete(self, pose):
        dists = np.linalg.norm(pose[:2] - self.center_line[:, :2], axis=1) # last 20 points.
        t_guess = self.s_track[np.argmin(dists)] / self.s_track[-1]
        t_point = optimize.fmin(dist_to_p, x0=t_guess, args=(self.tck, pose[:2]), disp=False)[0]
        
        done = False
        if t_point > 0.99 and self.current_time > 5: done = True
        if self.current_time > 150: 
            print("Time limit reached")
            done = True

        return done, t_point
        

    def check_vehicle_collision(self, pose):
        rotation_mtx = np.array([[np.cos(pose[2]), -np.sin(pose[2])], [np.sin(pose[2]), np.cos(pose[2])]])

        pts = np.array([[self.params['length']/2, self.params['width']/2], 
                        [self.params['length']/2, -self.params['width']/2], 
                        [-self.params['length']/2, self.params['width']/2], 
                        [-self.params['length']/2, -self.params['width']/2]])
        pts = np.matmul(pts, rotation_mtx.T) + pose[0:2]

        for i in range(4):
            if self.scan_simulator.check_location(pts[i, :]):
                return True

        return False

    

    def reset(self, poses):
        """
        Reset the gym environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        # reset counters and data members
        self.current_time = 0.0

        self.dynamics_simulator.reset(poses)

        # get no input observations
        action = np.zeros(2)
        obs, done = self.step(action)

        self.lap_number += 1
        
        return obs, done


def dist_to_p(t_glob: np.ndarray, path: list, p: np.ndarray):
    s = interpolate.splev(t_glob, path, ext=3)
    s = np.concatenate(s)
    return spatial.distance.euclidean(p, s)
