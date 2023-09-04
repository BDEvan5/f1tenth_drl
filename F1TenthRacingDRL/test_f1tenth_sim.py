import yaml
from argparse import Namespace
import os 
import numpy as np

from F1TenthRacingDRL.f1tenth_sim import F1TenthSim
from F1TenthRacingDRL.Utils.utils import *
from F1TenthRacingDRL.Planners.PurePursuit import RacingPurePursuit
import torch



def run_simulation_loop_steps(env, planner, steps, steps_per_action=10):
    observation, done = env.reset(poses=np.array([0, 0, 0]))
    
    for i in range(steps):
        action = planner.plan(observation)
        
        mini_i = steps_per_action
        while mini_i > 0: #TODO: move this loop to inside the simulator.
            observation, done = env.step(action[None, :])
            mini_i -= 1
        
            if done:
                planner.done_callback(observation)
                observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))
                break
                  
        if RENDER_ENV: env.render('human_fast')
        
def run_simulation_loop_laps(env, planner, n_laps, n_sim_steps=10):
    for lap in range(n_laps):
        observation, done = env.reset(poses=np.array([0, 0, 0]))
        while not done:
            action = planner.plan(observation)
            
            mini_i = n_sim_steps
            while mini_i > 0 and not done:
                observation, done = env.step(action)
                mini_i -= 1
    
        planner.done_callback(observation)
    
def seed_randomness(run_dict):
    random_seed = run_dict.random_seed + 10 * run_dict.n
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(random_seed)
    

def run_testing_batch():
    conf = load_conf("config_file")
    
    # seed_randomness(12345)
    print(f"Running experiment")

    run_dict = {"map_name": "aut", "run_name": "test", "n": 0, "random_seed": 12345, "n_test_laps": 1, "path": "TestSim", "racing_line": True, "pp_speed_mode": "raceline", "max_speed": 8}
    run_dict = Namespace(**run_dict)

    env = F1TenthSim(run_dict.map_name, seed=12345)
    print("Testing")
    planner = RacingPurePursuit(conf, run_dict, False)

    run_simulation_loop_laps(env, planner, run_dict.n_test_laps, 5)
        

def run_general_test_batch(experiment):
    run_list = setup_run_list(experiment, new_run=False)
    conf = load_conf("config_file")
    map_list = ["aut", "esp", "gbr", "mco"]
    
    for i, run_dict in enumerate(run_list):
        print(f"Running experiment: {i}")
        print(f"RunName: {run_dict.run_name}")
        for m in range(len(map_list)):
            print(f"Testing on map: {map_list[m]}")
            run_dict.map_name = map_list[m]
            env = F110Env(map=run_dict.map_name, num_agents=1)
            print("Testing")
            planner = select_test_agent(conf, run_dict)
            run_simulation_loop_laps(env, planner, run_dict.n_test_laps, 10)
        

    
def main():
    # experiment = "Experiment"
    # experiment = "main"
    
    # run_training_batch(experiment)

    run_testing_batch()

    # run_general_test_batch(experiment)


def run_pp_tests():
    experiment = "ConfigPP"

    # run_testing_batch(experiment)

    
if __name__ == "__main__":
    main()
    # run_pp_tests()
    # run_general_test_batch()

