import numpy as np

from RacingDRL.Planners.StdTrack import StdTrack



# rewards functions
class ProgressReward:
    def __init__(self, track: StdTrack) -> None:
        self.track = track

    def __call__(self, observation, prev_obs):
        if prev_obs is None: return 0

        if observation['lap_counts'][0]:
            return 1  # complete
        if observation['collisions'][0]:
            return -1 # crash
        
        position = np.array([observation['poses_x'][0], observation['poses_y'][0]])
        prev_position = np.array([prev_obs['poses_x'][0], prev_obs['poses_y'][0]])

        s = self.track.calculate_progress(prev_position)
        ss = self.track.calculate_progress(position)
        reward = (ss - s) / self.track.total_s
        if abs(reward) > 0.5: # happens at end of eps
            return 0.001 # assume positive progress near end


        return reward 
    
