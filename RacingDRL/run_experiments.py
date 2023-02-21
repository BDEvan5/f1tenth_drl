import yaml
from argparse import Namespace
import os 

from RacingDRL.f1tenth_gym import F110Env
from RacingDRL.Planners.AgentTrainer import AgentTrainer

def generate_test_name(file_name):
    n = 1
    while os.path.exists(f"Data/{file_name}_{n}"):
        n += 1
    os.mkdir(f"Data/{file_name}_{n}")
    return file_name + f"_{n}"

def setup_run_list(experiment_file):
    full_path =  "experiments/" + experiment_file + '.yaml'
    with open(full_path) as file:
        experiment_dict = yaml.load(file, Loader=yaml.FullLoader)
        
    test_name = generate_test_name(experiment_file)

    run_list = []
    for rep in range(experiment_dict['n_repeats']):
        for run in experiment_dict['runs']:
            # base is to copy everything from the original
            for key in experiment_dict.keys():
                if key not in run.keys() and key != "runs":
                    run[key] = experiment_dict[key]

            # only have to add what isn't already there
            set_n = run['set_n']
            max_speed = run['max_speed']
            run["n"] = rep
            run['run_name'] = f"{run['planner_type']}_{run['algorithm']}_{run['state_vector']}_{run['map_name']}_{max_speed}_{set_n}_{rep}"
            run['path'] = f"{test_name}/"

            run_list.append(Namespace(**run))

    return run_list

def create_planner(run_dict):
    if run_dict.planner_type == "Agent":
        planner = AgentTrainer(run_dict)
    elif run_dict.planner_type == "pure_puresuit":
        planner = PurePursuit(run_dict)
    else: raise ValueError(f"Planner type {run_dict.planner_type} not recognised")
    
    return planner

RENDER_ENV = False

def run_simulation_loop(env, planner, run_dict):
    observation = env.reset()
    
    for i in range(run_dict.training_steps):
        action = planner.get_action(observation)
        observation, reward, done, info = env.step(action)
        
        if done:
            planner.done_callback(observation)
            observation = env.reset()
                  
        if RENDER_ENV: env.render('human_fast')
        
    
def run_training_batch(experiment):
    run_list = setup_run_list(experiment)
    
    for i, run_dict in enumerate(run_list):
        print(f"Running experiment: {i}")
        print(f"RunName: {run_dict.run_name}")

        env = F110Env(map=run_dict.map_name)
        planner = AgentTrainer(run_dict)
        
        run_simulation_loop(env, planner, run_dict)
        
        planner = AgentTester(run_dict)
        run_simulation_loop(env, planner, run_dict)
        
        


if __name__ == "__main__":
    experiment = "main"
    


