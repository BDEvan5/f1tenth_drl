"""
Partial code source: https://github.com/f1tenth/f1tenth_gym
Example waypoint_follow.py from f1tenth_gym
Specific function used:
- nearest_point_on_trajectory_py2
- first_point_on_trajectory_intersecting_circle
- get_actuation

Adjustments have been made

"""

import numpy as np
from numba import njit
import csv
import os
from matplotlib import pyplot as plt
from FTenthRacingDRL.Planners.TrackLine import TrackLine
from FTenthRacingDRL.FitChecking.DataStateHistory import DataStateHistory



class PurePursuitDataGen:
    def __init__(self, conf, experiment):
        self.conf = conf

        self.racing_line = experiment.racing_line
        self.speed_mode = experiment.pp_speed_mode
        self.max_speed = experiment.max_speed
        self.track_line = TrackLine(experiment.map_name, experiment.racing_line, False)
        self.lookahead = experiment.lookahead

        self.v_min_plan = conf.v_min_plan
        self.wheelbase =  conf.l_f + conf.l_r
        self.max_steer = conf.max_steer
        
        self.data_history = DataStateHistory(experiment)

        self.counter = 0

    def plan(self, obs):
        position = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        theta = obs['poses_theta'][0]
        
        # lookahead = self.lookahead
        lookahead = 1 + 0.6 * obs['linear_vels_x'][0] / 8
        lookahead_point = self.track_line.get_lookahead_point(position, lookahead)

        if obs['linear_vels_x'][0] < self.v_min_plan:
            return np.array([0.0, 4])

        speed_raceline, steering_angle = get_actuation(theta, lookahead_point, position, lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        speed = speed_raceline 
        speed = min(speed, self.max_speed) # cap the speed

        action = np.array([steering_angle, speed])
        self.data_history.add_memory_entry(obs, action)

        return action

    def done_callback(self, final_obs):
        self.data_history.add_memory_entry(final_obs, np.array([0, 0]))
        self.data_history.save_history()
        
        progress = self.track_line.calculate_progress_percent([final_obs['poses_x'][0], final_obs['poses_y'][0]]) * 100
        
        print(f"Test lap complete --> Time: {final_obs['lap_times'][0]:.2f}, Colission: {bool(final_obs['collisions'][0])}, Lap p: {progress:.1f}%")



@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

