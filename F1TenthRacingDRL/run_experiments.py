import yaml
from argparse import Namespace
import os 
import numpy as np

from F1TenthRacingDRL.f1tenth_sim import F1TenthSim
from F1TenthRacingDRL.Planners.AgentTrainer import AgentTrainer
from F1TenthRacingDRL.Planners.AgentTester import AgentTester
from F1TenthRacingDRL.Utils.utils import *
from F1TenthRacingDRL.Planners.PurePursuit import RacingPurePursuit
import torch



RENDER_ENV = False
# RENDER_ENV = True

        
def  select_test_agent(conf, run_dict):
    if run_dict.planner_type == "AgentOff":
        planner = AgentTester(run_dict, conf)
    elif run_dict.planner_type == "PurePursuit":
        planner = RacingPurePursuit(conf, run_dict, False)
    else:
        raise ValueError(f"Planner type not recognised: {run_dict.planner_type}")    
    
    return planner

def run_simulation_loop_steps(env, planner, steps):
    observation, done = env.reset(poses=np.array([0, 0, 0]))
    
    for i in range(steps):
        action = planner.plan(observation)
        observation, done = env.step(action)
        if done:
            planner.done_callback(observation)
            observation, done = env.reset(poses=np.array([0, 0, 0]))
                  
def run_simulation_loop_laps(env, planner, n_laps):
    for lap in range(n_laps):
        observation, done = env.reset(poses=np.array([0, 0, 0]))
        while not done:
            action = planner.plan(observation)
            observation, done = env.step(action)
    
        planner.done_callback(observation)
    
def seed_randomness(run_dict):
    random_seed = run_dict.random_seed + 10 * run_dict.n
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(random_seed)
    
def run_training_batch(experiment):
    # run_list = setup_run_list(experiment, new_run=False)
    run_list = setup_run_list(experiment)
    conf = load_conf("config_file")
    
    for i, run_dict in enumerate(run_list):
        seed_randomness(run_dict)
        print(f"Running experiment: {i}")
        print(f"RunName: {run_dict.run_name}")

        env = F1TenthSim(run_dict, False)
        # planner = AgentTrainer(run_dict, conf)
        
        # print("Training")
        # run_simulation_loop_steps(env, planner, run_dict.training_steps)
        
        print("Testing")
        planner = AgentTester(run_dict, conf)
        run_simulation_loop_laps(env, planner, run_dict.n_test_laps)
        
    
def run_testing_batch(experiment, n_sim_steps=10):
    # run_list = setup_run_list(experiment, new_run=True)
    run_list = setup_run_list(experiment, new_run=False)
    conf = load_conf("config_file")
    
    for i, run_dict in enumerate(run_list):
        seed_randomness(run_dict)
        print(f"Running experiment: {i}")
        print(f"RunName: {run_dict.run_name}")

        env = F110Env(map=run_dict.map_name, num_agents=1)
        print("Testing")
        planner = select_test_agent(conf, run_dict)
        run_simulation_loop_laps(env, planner, run_dict.n_test_laps, n_sim_steps)
        

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
    experiment = "main"
    
    run_training_batch(experiment)

    # run_testing_batch(experiment, 10)

    # run_general_test_batch(experiment)


def run_pp_tests():
    experiment = "ConfigPP"

    run_testing_batch(experiment)


    # run_pp_tests()
    # run_general_test_batch()

if __name__ == "__main__":
    main()
    # run_pp_tests()
    # run_general_test_batch()

