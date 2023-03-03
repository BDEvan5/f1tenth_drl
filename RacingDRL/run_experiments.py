import yaml
from argparse import Namespace
import os 
import numpy as np

from RacingDRL.f1tenth_gym import F110Env
from RacingDRL.Planners.AgentTrainer import AgentTrainer
from RacingDRL.Planners.AgentTester import AgentTester
from RacingDRL.Utils.utils import *
from RacingDRL.Planners.PurePursuit import PurePursuit


RENDER_ENV = False
# RENDER_ENV = True

        
def select_test_agent(conf, run_dict):
    if run_dict.planner_type == "AgentOff":
        planner = AgentTester(run_dict, conf)
    elif run_dict.planner_type == "PurePursuit":
        planner = PurePursuit(conf, run_dict, False)
    else:
        raise ValueError(f"Planner type not recognised: {run_dict.planner_type}")    
    
    return planner

def run_simulation_loop_steps(env, planner, steps):
    observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))
    
    for i in range(steps):
        action = planner.plan(observation)
        
        mini_i = 10
        while mini_i > 0:
            observation, reward, done, info = env.step(action[None, :])
            mini_i -= 1
        
            if done:
                planner.done_callback(observation)
                observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))
                break
                  
        if RENDER_ENV: env.render('human_fast')
        
def run_simulation_loop_laps(env, planner, n_laps):
    observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))
    
    for lap in range(n_laps):
        while not done:
            action = planner.plan(observation)
            
            mini_i = 10
            while mini_i > 0 and not done:
                observation, reward, done, info = env.step(action[None, :])
                mini_i -= 1

            if RENDER_ENV: env.render('human')
            
            # if RENDER_ENV: env.render('human_fast')
    
        planner.done_callback(observation)
        observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))
                  
        
    
def run_training_batch(experiment):
    # run_list = setup_run_list(experiment, new_run=False)
    run_list = setup_run_list(experiment)
    conf = load_conf("config_file")
    
    for i, run_dict in enumerate(run_list):
        print(f"Running experiment: {i}")
        print(f"RunName: {run_dict.run_name}")

        env = F110Env(map=run_dict.map_name, num_agents=1)
        planner = AgentTrainer(run_dict, conf)
        
        print("Training")
        run_simulation_loop_steps(env, planner, run_dict.training_steps)
        
        print("Testing")
        planner = AgentTester(run_dict, conf)
        run_simulation_loop_laps(env, planner, run_dict.n_test_laps)
        env.__del__()
        
    
def run_testing_batch(experiment):
    # run_list = setup_run_list(experiment, new_run=True)
    run_list = setup_run_list(experiment, new_run=False)
    conf = load_conf("config_file")
    
    for i, run_dict in enumerate(run_list):
        print(f"Running experiment: {i}")
        print(f"RunName: {run_dict.run_name}")

        env = F110Env(map=run_dict.map_name, num_agents=1)
        print("Testing")
        planner = select_test_agent(conf, run_dict)
        run_simulation_loop_laps(env, planner, run_dict.n_test_laps)
        
        
import cProfile
import pstats
import io
from pstats import SortKey
    
def main():
    # experiment = "GameAlgorithms"
    experiment = "TrajectoryNumPoints"
    # experiment = "main"
    # experiment = "testPP"
    run_training_batch(experiment)
    # run_testing_batch(experiment)
    
def profile():
    with cProfile.Profile(builtins=False) as pr:
        main()
        
        with open("Data/Profiling/main.prof", "w") as f:
            ps = pstats.Stats(pr, stream=f)
            ps.strip_dirs()
            ps.sort_stats('cumtime')
            ps.print_stats()
            
        with open("Data/Profiling/main_total.prof", "w") as f:
            ps = pstats.Stats(pr, stream=f)
            ps.strip_dirs()
            ps.sort_stats('tottime')
            ps.print_stats()
    
from RacingDRL.DataTools.TrajAnalysis.GenerateTrajectoryAnalysis import analyse_folder
    
if __name__ == "__main__":
    main()
    # analyse_folder()
    # profile()


