from F1TenthRacingDRL.Utils.utils import init_file_struct, save_run_config
from F1TenthRacingDRL.LearningAlgorithms.create_agent import create_train_agent
import numpy as np
from F1TenthRacingDRL.Planners.Architectures import select_architecture
from F1TenthRacingDRL.Utils.TrainHistory import TrainHistory
from F1TenthRacingDRL.Planners.RewardSignals import select_reward_function
from F1TenthRacingDRL.Planners.TrackLine import TrackLine


class AgentTrainer: 
    def __init__(self, run, conf):
        self.run, self.conf = run, conf
        self.name = run.run_name
        self.path = conf.vehicle_path + run.path + run.run_name + "/"
        init_file_struct(self.path)
        save_run_config(run.__dict__, self.path)

        self.v_min_plan =  conf.v_min_plan

        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None
        self.std_track = TrackLine(run.map_name, False)
        # self.reward_generator = ProgressReward(self.std_track)
        # self.reward_generator = CrossTrackHeadReward(self.std_track, conf)
        # self.reward_generator = TALearningReward(conf, run)
        self.reward_generator = select_reward_function(run, conf, self.std_track)
        self.max_lap_progress = 0

        self.architecture = select_architecture(run, conf)
        self.agent = create_train_agent(run, self.architecture.state_space)
        self.t_his = TrainHistory(self.path)

    def plan(self, obs):
        vehicle_state = obs["vehicle_state"]
        progress = self.std_track.calculate_progress_percent(vehicle_state[:2]) * 100
        self.max_lap_progress = max(self.max_lap_progress, progress)
        
        nn_state = self.architecture.transform_obs(obs)
        
        self.add_memory_entry(obs, nn_state)
        self.state = obs
            
        if vehicle_state[3] < self.v_min_plan:
            self.action = np.array([0, 2])
            return self.action

        self.nn_state = nn_state 
        self.nn_act = self.agent.act(self.nn_state)
        self.action = self.architecture.transform_action(self.nn_act)
        
        self.agent.train()

        return self.action 

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.nn_state is not None:
            reward = self.reward_generator(s_prime, self.state, self.action)
            self.t_his.add_step_data(reward)

            self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, False)

    def done_callback(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime = self.architecture.transform_obs(s_prime)
        reward = self.reward_generator(s_prime, self.state, self.action)
        state_prime = s_prime['vehicle_state']
        progress = self.std_track.calculate_progress_percent(state_prime[:2]) * 100
        self.max_lap_progress = max(self.max_lap_progress, progress)
        
        # print(self.t_his.reward_list)
        self.t_his.lap_done(reward, self.max_lap_progress, False)
        print(f"Episode: {self.t_his.ptr}, Step: {self.t_his.t_counter}, Lap p: {self.max_lap_progress:.1f}%, Reward: {self.t_his.rewards[self.t_his.ptr-1]:.2f}")

        if self.nn_state is None:
            print(f"Crashed on first step: RETURNING")
            return
        
        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, True)
        self.nn_state = None
        self.state = None
        self.max_lap_progress = 0

        self.save_training_data()

    def save_training_data(self):
        self.t_his.print_update(True)
        self.t_his.save_csv_data()
        self.agent.save(self.name, self.path)