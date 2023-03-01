import numpy as np
import os

class VehicleStateHistory:
    def __init__(self, run, folder):
        self.vehicle_name = run.run_name
        self.path = "Data/" + run.path + run.run_name + "/" + folder
        if os.path.exists(self.path) == False:
            os.mkdir(self.path)
        self.states = []
        self.actions = []
    
        self.lap_n = 0
    
    def add_memory_entry(self, obs, action):
        state = obs['full_states'][0]

        self.states.append(state)
        self.actions.append(action)
    
    def save_history(self, test_map=None):
        states = np.array(self.states)
        actions = np.array(self.actions)

        lap_history = np.concatenate((states, actions), axis=1)
        
        print(f"Last state: {states[-1]}")

        if test_map is None:
            np.save(self.path + f"Lap_{self.lap_n}_history_{self.vehicle_name}.npy", lap_history)
        else:
            np.save(self.path + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{test_map}.npy", lap_history)

        self.states = []
        self.actions = []
        self.lap_n += 1
        
    


if __name__ == "__main__":
    pass