
import numpy as np
from F1TenthRacingDRL.Utils.utils import init_file_struct, calculate_speed
from numba import njit
import csv
import os
from matplotlib import pyplot as plt
from F1TenthRacingDRL.Planners.TrackLine import TrackLine
from F1TenthRacingDRL.Planners.VehicleStateHistory import VehicleStateHistory



class RacingPurePursuit:
    def __init__(self, conf, run, init=True):
        self.name = run.run_name
        path = os.getcwd() + f"/Data/" + run.path  + self.name + "/"
        if not os.path.exists(path):
            os.mkdir(path)
        elif init:
            init_file_struct(path)
            
        self.conf = conf
        self.run = run

        self.max_speed = 8
        self.track_line = TrackLine(run.map_name, True, False)

        self.v_min_plan = conf.v_min_plan
        self.wheelbase =  conf.l_f + conf.l_r
        self.max_steer = conf.max_steer
        
        self.counter = 0

    def plan(self, obs):
        position = obs["vehicle_state"][0:2]
        theta = obs["vehicle_state"][4]

        lookahead = 0.4 + 0.16 * obs["vehicle_state"][3]
        lookahead_point = self.track_line.get_lookahead_point(position, lookahead)

        if obs["vehicle_state"][3] < self.v_min_plan:
            return np.array([0.0, 4])

        speed_raceline, steering_angle = get_actuation(theta, lookahead_point, position, lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
            
        speed = min(speed_raceline, self.max_speed) # cap the speed
        action = np.array([steering_angle, speed])
        self.vehicle_state_history.add_memory_entry(obs, action)

        return action

    def done_callback(self, final_obs):
        pass



@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

