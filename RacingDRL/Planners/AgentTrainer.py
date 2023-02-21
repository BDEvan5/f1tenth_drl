from RacingDRL.Utils.utils import init_file_struct
from RacingDRL.LearningAlgorithms.create_agent import create_agent
import numpy as np
from RacingDRL.Planners.Architectures import ArchEndToEnd, ArchHybrid, ArchPathFollower
from RacingDRL.Utils.HistoryStructs import TrainHistory


class AgentTrainer: 
    def __init__(self, run, conf):
        self.run, self.conf = run, conf
        self.name = run.run_name
        self.path = conf.vehicle_path + run.path + run.run_name 
        init_file_struct(self.path)

        self.v_min_plan =  conf.v_min_plan

        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None

        self.agent = create_agent(run)
        if run.state_vector == "end_to_end":
            self.architecture = ArchEndToEnd(run, conf)

        self.t_his = TrainHistory(run, conf)

    def plan(self, obs):
        nn_state = self.architecture.transform_obs(obs)
        self.add_memory_entry(obs, nn_state)
            
        if obs['state'][3] < self.v_min_plan:
            self.action = np.array([0, 2])
            return self.action

        self.nn_state = nn_state 
        self.nn_act = self.agent.act(self.nn_state)

        # self.architecture.transform_obs(obs) #! to ensure correct PP actions
        self.action = self.architecture.transform_action(self.nn_act)

        return self.action 

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.nn_state is not None:
            self.t_his.add_step_data(s_prime['reward'])

            self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, s_prime['reward'], False)

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime = self.architecture.transform_obs(s_prime)

        self.t_his.lap_done(s_prime['reward'], s_prime['progress'], False)
        if self.nn_state is None:
            print(f"Crashed on first step: RETURNING")
            return
        
        self.agent.save(self.path)

        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, s_prime['reward'], True)
        self.nn_state = None

    def lap_complete(self):
        pass

    def save_training_data(self):
        self.t_his.print_update(True)
        self.t_his.save_csv_data()
        self.agent.save(self.path)