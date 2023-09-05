from F1TenthRacingDRL.Utils.utils import init_file_struct
from F1TenthRacingDRL.LearningAlgorithms.create_agent import create_test_agent
import numpy as np
from F1TenthRacingDRL.Planners.Architectures import select_architecture
from F1TenthRacingDRL.Planners.TrackLine import TrackLine
from F1TenthRacingDRL.Planners.VehicleStateHistory import VehicleStateHistory


class AgentTester: 
    def __init__(self, run, conf):
        self.run, self.conf = run, conf
        self.name = run.run_name
        self.path = conf.vehicle_path + run.path + run.run_name + "/"

        self.v_min_plan =  conf.v_min_plan

        self.std_track = TrackLine(run.map_name, False)
        self.architecture = select_architecture(run, conf)

        self.agent = create_test_agent(self.name, self.path, run)
        
    def plan(self, obs):
        nn_state = self.architecture.transform_obs(obs)
        
        if obs['vehicle_state'][3] < self.v_min_plan:
            return np.array([0, 2])

        self.nn_act = self.agent.act(nn_state)
        self.action = self.architecture.transform_action(self.nn_act)
        
        
        return self.action 

    def done_callback(self, s_prime):
        """
        To be called when ep is done.
        """
        progress = self.std_track.calculate_progress_percent(s_prime['vehicle_state'][:2]) * 100
        

