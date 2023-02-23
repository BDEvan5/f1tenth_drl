import os, shutil, yaml
from argparse import Namespace

def init_file_struct(path):
    if os.path.exists(path):
        try:
            os.rmdir(path)
        except:
            shutil.rmtree(path)
    os.mkdir(path)

    
def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
       
       
def load_conf(fname):
    full_path =  "experiments/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf 


def generate_test_name(file_name):
    n = 1
    while os.path.exists(f"Data/{file_name}_{n}"):
        n += 1
    os.mkdir(f"Data/{file_name}_{n}")
    return file_name + f"_{n}"

def setup_run_list(experiment_file, new_run=True):
    full_path =  "experiments/" + experiment_file + '.yaml'
    with open(full_path) as file:
        experiment_dict = yaml.load(file, Loader=yaml.FullLoader)
        
    test_name = generate_test_name(experiment_file)
    if not new_run: test_name = test_name.split("_")[:-1].join("_") + (test_name.split("_")[-1]) - 1

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
