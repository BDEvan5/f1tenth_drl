import numpy as np

from F1TenthRacingDRL.Planners.TrackLine import TrackLine
from F1TenthRacingDRL.Planners.PurePursuit import RacingPurePursuit


def select_reward_function(run, conf, std_track):
    reward = run.run_name.split("_")[4]
    if reward ==  "TAL":
        return TALearningReward(conf, run)
    elif reward == "Progress":
        return ProgressReward(std_track)
    elif reward == "Cth":
        return CrossTrackHeadReward(std_track, conf)
    elif reward == "Speed":
        return SpeedReward(std_track, conf)
    else:
        raise ValueError(f"Reward function not recognised: {reward}")


# rewards functions
class ProgressReward:
    def __init__(self, track: TrackLine) -> None:
        self.track = track

    def __call__(self, observation, prev_obs, action):
        if prev_obs is None: return 0

        if observation['lap_counts'][0]:
            return 1  # complete
        if observation['collisions'][0]:
            return -1 # crash
        
        position = np.array([observation['poses_x'][0], observation['poses_y'][0]])
        prev_position = np.array([prev_obs['poses_x'][0], prev_obs['poses_y'][0]])

        s = self.track.calculate_progress_percent(prev_position)
        ss = self.track.calculate_progress_percent(position)
        reward = ss - s
        if abs(reward) > 0.5: # happens at end of eps
            return 0.001 # assume positive progress near end

        reward *= 10

        return reward 
    

class CrossTrackHeadReward:
    def __init__(self, track: TrackLine, conf):
        self.track = track
        self.r_veloctiy = 1
        self.r_distance = 1
        self.max_v = conf.max_v # used for scaling.

    def __call__(self, observation, prev_obs, pre_action):
        if observation['lap_counts'][0]:
            return 1  # complete
        if observation['collisions'][0]:
            return -1 # crash

        position = np.array([observation['poses_x'][0], observation['poses_y'][0]])
        theta = observation['poses_theta'][0]
        speed = observation['linear_vels_x'][0]
        heading, distance = self.track.get_cross_track_heading(position)

        d_heading = abs(robust_angle_difference_rad(heading, theta))
        r_heading  = np.cos(d_heading)  * self.r_veloctiy # velocity
        r_heading *= (speed / self.max_v)

        r_distance = distance * self.r_distance 

        reward = r_heading - r_distance
        reward = max(reward, 0)
        return reward

def robust_angle_difference_rad(x, y):
    """Returns the difference between two angles in RADIANS
    r = x - y"""
    return np.arctan2(np.sin(x-y), np.cos(x-y))


class SpeedReward:
    def __init__(self, track: TrackLine, conf):
        self.max_v = conf.max_v # used for scaling.

    def __call__(self, observation, prev_obs, pre_action):
        if observation['lap_counts'][0]:
            return 1  # complete
        if observation['collisions'][0]:
            return -1 # crash

        # theta = observation['poses_theta'][0]
        speed = observation['linear_vels_x'][0]

        reward = (speed / self.max_v) ** 2
        reward = reward * 0.5 # scale to a reasonable range.

        return reward

def robust_angle_difference_rad(x, y):
    """Returns the difference between two angles in RADIANS
    r = x - y"""
    return np.arctan2(np.sin(x-y), np.cos(x-y))





class TALearningReward:
    def __init__(self, conf, run):
        run.pp_speed_mode = "racing_line"
        run.racing_line = True
        # self.lookahead = 1.5
        self.pp = RacingPurePursuit(conf, run, False) 

        self.beta_c = 0.4
        self.beta_steer_weight = 0.4
        self.beta_velocity_weight = 0.4

        self.max_steer_diff = 0.8
        self.max_velocity_diff = 2.0
        
    def __call__(self, observation, prev_obs, action):
        if prev_obs is None: return 0

        if observation['lap_complete']:
            return 1  # complete
        if observation['collision']:
            return -1 # crash
        
        pp_act = self.pp.plan(prev_obs)

        steer_reward =  (abs(pp_act[0] - action[0]) / self.max_steer_diff)  * self.beta_steer_weight

        throttle_reward =   (abs(pp_act[1] - action[1]) / self.max_velocity_diff) * self.beta_velocity_weight

        reward = self.beta_c - steer_reward - throttle_reward
        reward = max(reward, 0) # limit at 0

        reward *= 0.5

        return reward



