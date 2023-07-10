import glob
import numpy as np 
from argparse import Namespace
import numpy as np

from FTenthRacingDRL.f1tenth_gym import F110Env
from FTenthRacingDRL.f1tenth_gym import RaceCar
from FTenthRacingDRL.Utils.utils import *
from FTenthRacingDRL.FitChecking.PurePursuitDataGen import PurePursuitDataGen


def run_simulation_loop_laps(env, planner, n_laps, n_sim_steps=10):
    observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))
    
    for lap in range(n_laps):
        while not done:
            action = planner.plan(observation)
            
            mini_i = 10
            while mini_i > 0 and not done:
                observation, reward, done, info = env.step(action[None, :])
                mini_i -= 1

        planner.done_callback(observation)
        observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))


def generate_pp_data():
    experiment_name = "GenerateDataSet"
    with open(f"experiments/{experiment_name}" + '.yaml') as file:
        experiment_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = load_conf("config_file")
        
    experiment_dict['folder'] = f"Data/{experiment_name}_{experiment_dict['set_n']}/"
    experiment = Namespace(**experiment_dict)
    
    if not os.path.exists(experiment.folder): os.mkdir(experiment.folder)
    
    for m in range(len(experiment.map_list)):
        print(f"Testing on map: {experiment.map_list[m]}")
        experiment.map_name = experiment.map_list[m]
        
        planner = PurePursuitDataGen(conf, experiment)
        
        env = F110Env(map=experiment.map_list[m], num_agents=1)
        RaceCar.scan_simulator = None
        env.sim.agents[0] = RaceCar(env.sim.params, env.sim.seed, num_beams=60, time_step=env.sim.time_step, integrator=env.sim.integrator)
        env.sim.set_map(env.map_path, env.map_ext)
        run_simulation_loop_laps(env, planner, experiment.n_test_laps)
        
        
if __name__ == "__main__":
    generate_pp_data()