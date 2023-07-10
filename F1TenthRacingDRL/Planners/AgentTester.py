from FTenthRacingDRL.Utils.utils import init_file_struct
from FTenthRacingDRL.LearningAlgorithms.create_agent import create_test_agent
import numpy as np
from FTenthRacingDRL.Planners.Architectures import select_architecture
from FTenthRacingDRL.Planners.TrackLine import TrackLine
from FTenthRacingDRL.Planners.VehicleStateHistory import VehicleStateHistory


class AgentTester: 
    def __init__(self, run, conf):
        self.run, self.conf = run, conf
        self.name = run.run_name
        self.path = conf.vehicle_path + run.path + run.run_name + "/"

        self.v_min_plan =  conf.v_min_plan

        self.std_track = TrackLine(run.map_name, False)
        self.architecture = select_architecture(run, conf)

        self.agent = create_test_agent(self.name, self.path, run)
        
        self.vehicle_state_history = VehicleStateHistory(run, f"Testing{run.map_name.upper()}/")
#capitalise

    def plan(self, obs):
        nn_state = self.architecture.transform_obs(obs)
        
        if obs['linear_vels_x'][0] < self.v_min_plan:
            return np.array([0, 2])

        self.nn_act = self.agent.act(nn_state)
        self.action = self.architecture.transform_action(self.nn_act)
        
        self.vehicle_state_history.add_memory_entry(obs, self.action)
        
        return self.action 

    def done_callback(self, s_prime):
        """
        To be called when ep is done.
        """
        progress = self.std_track.calculate_progress_percent([s_prime['poses_x'][0], s_prime['poses_y'][0]]) * 100
        
        print(f"Test lap complete --> Time: {s_prime['lap_times'][0]:.2f}, Colission: {bool(s_prime['collisions'][0])}, Lap p: {progress:.1f}%")

        self.vehicle_state_history.save_history()

