import numpy as np
import os

class DataStateHistory:
    def __init__(self, experiment):
        self.path = experiment.folder + "RawData/"
        if not os.path.exists(self.path): os.mkdir(self.path)
        self.vehicle_name = f"{experiment.planner_type}_{experiment.map_name}_{experiment.id_name}_{experiment.set_n}"
        self.states = []
        self.actions = []
        self.scans = []
    
        self.lap_n = 0
    
    def add_memory_entry(self, obs, action):
        state = obs['full_states'][0]

        self.states.append(state)
        self.actions.append(action)
        self.scans.append(obs['scans'][0])
    
    def save_history(self):
        states = np.array(self.states)
        actions = np.array(self.actions)

        lap_history = np.concatenate((states, actions), axis=1)
        np.save(self.path + f"{self.vehicle_name}_Lap_{self.lap_n}_history.npy", lap_history)
        
        scans = np.array(self.scans)
        np.save(self.path + f"{self.vehicle_name}_Lap_{self.lap_n}_scans.npy", scans)
        
        self.states = []
        self.actions = []
        self.scans = []
        self.lap_n += 1
        
        """
            Single Track Dynamic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
                x6: yaw rate
                x7: slip angle at vehicle center
        
        """


if __name__ == "__main__":
    pass