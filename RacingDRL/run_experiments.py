import yaml
from argparse import Namespace
import os 
import numpy as np

from RacingDRL.f1tenth_gym import F110Env
from RacingDRL.Planners.AgentTrainer import AgentTrainer
from RacingDRL.Planners.AgentTester import AgentTester
from RacingDRL.Utils.utils import *
from RacingDRL.Planners.PurePursuit import PurePursuit
import glob

from RacingDRL.DataTools.TrajAnalysis.GenerateTrajectoryAnalysis import analyse_folder


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
        
def run_simulation_loop_laps(env, planner, n_laps, n_sim_steps=10):
    observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))
    
    for lap in range(n_laps):
        while not done:
            action = planner.plan(observation)
            
            mini_i = 1
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
        
    
def run_testing_batch(experiment, n_sim_steps=10):
    # run_list = setup_run_list(experiment, new_run=True)
    run_list = setup_run_list(experiment, new_run=False)
    conf = load_conf("config_file")
    
    for i, run_dict in enumerate(run_list):
        print(f"Running experiment: {i}")
        print(f"RunName: {run_dict.run_name}")

        env = F110Env(map=run_dict.map_name, num_agents=1)
        print("Testing")
        planner = select_test_agent(conf, run_dict)
        run_simulation_loop_laps(env, planner, run_dict.n_test_laps, n_sim_steps)
        

def run_general_test_batch():
    map_list = ["aut", "esp", "gbr", "mco"]
    folder = "TrajectoryNumPoints_3/"
    vehicles = glob.glob("Data/" + folder + "*/")
    print(vehicles)
    n_test_laps = 3
    
    conf = load_conf("config_file")
    
    for v in range(len(vehicles)):
        print(f"Testing Vehicle: {vehicles[v]}")
        for m in range(len(map_list)):
            print(f"Testing on map: {map_list[m]}")
            run_dict = load_run_dict(vehicles[v] + "TrainingConfig_record.yaml")
            run_dict.run_name = vehicles[v].split("/")[-2]
            run_dict.path = folder
            run_dict.map_name = map_list[m]
            
            env = F110Env(map=map_list[m], num_agents=1)
            planner = AgentTester(run_dict, conf)
            run_simulation_loop_laps(env, planner, n_test_laps)
        
        
def run_pp_tests():
    experiment = "PurePursuitMaps"
    run_testing_batch(experiment, n_sim_steps=1)
     
      
    
def main():
    # experiment = "EndNumBeams"
    # experiment = "TrajectoryNumPoints"
    # experiment = "GameAblation"
    
    # experiment = "EndMaps"
    experiment = "TrajectoryMaps"
    # experiment = "GameMaps"
    
    # experiment = "main"
    # experiment = "EndSpeeds"
    
    run_training_batch(experiment)
    # run_testing_batch(experiment)


    
    
if __name__ == "__main__":
    main()
    # run_pp_tests()
  
    # run_general_test_batch()

