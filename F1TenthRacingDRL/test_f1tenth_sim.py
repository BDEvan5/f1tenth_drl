import yaml
from argparse import Namespace
import os 
import numpy as np

from F1TenthRacingDRL.f1tenth_sim import F1TenthSim
from F1TenthRacingDRL.Utils.utils import *
from F1TenthRacingDRL.Planners.PurePursuit import RacingPurePursuit
import torch


        
        
def run_simulation_loop_laps(env, planner, n_laps):
    for lap in range(n_laps):
        observation, done = env.reset(poses=np.array([0, 0, 0]))
        while not done:
            action = planner.plan(observation)
            observation, done = env.step(action)
    
        planner.done_callback(observation)
    

def run_testing_batch():
    conf = load_conf("config_file")
    
    # seed_randomness(12345)
    print(f"Running experiment")

    run_dict = {"map_name": "aut", "run_name": "SimTest", "n": 0, "random_seed": 12345, "n_test_laps": 20, "path": "TestSim", "n_sim_steps": 5}
    run_dict = Namespace(**run_dict)

    env = F1TenthSim(run_dict, True)
    planner = RacingPurePursuit(conf, run_dict, False)

    run_simulation_loop_laps(env, planner, run_dict.n_test_laps)
        

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

  
def run_profiling(function, name):
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    function()
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    with open(f"Data/profile_{name}.txt", "w") as f:
        ps.print_stats()
        f.write(s.getvalue())

    
if __name__ == "__main__":
    # main()
    run_profiling(main, "NewSim")
    # run_pp_tests()
    # run_general_test_batch()

