import os, shutil, yaml, csv
from argparse import Namespace
from numba import njit
import numpy as np
import cProfile
import pstats
import io
from pstats import SortKey

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

def load_run_dict(full_path):
    with open(full_path) as file:
        run_dict = yaml.load(file, Loader=yaml.FullLoader)

    run_dict = Namespace(**run_dict)

    return run_dict 


def generate_test_name(file_name):
    n = 1
    while os.path.exists(f"Data/{file_name}_{n}"):
        n += 1
    os.mkdir(f"Data/{file_name}_{n}")
    return file_name + f"_{n}"

def latest_test_name(file_name):
    n = 1
    while os.path.exists(f"Data/{file_name}_{n}"):
        n += 1
    return file_name + f"_{n-1}"

def setup_run_list(experiment_file, new_run=True):
    full_path =  "experiments/" + experiment_file + '.yaml'
    with open(full_path) as file:
        experiment_dict = yaml.load(file, Loader=yaml.FullLoader)
        
    set_n = experiment_dict['set_n']
    test_name = experiment_file + f"_{set_n}"
    if not os.path.exists(f"Data/{test_name}"):
        os.mkdir(f"Data/{test_name}")

    run_list = []
    for rep in range(experiment_dict['start_n'], experiment_dict['n_repeats']):
        for run in experiment_dict['runs']:
            # base is to copy everything from the original
            for key in experiment_dict.keys():
                if key not in run.keys() and key != "runs":
                    run[key] = experiment_dict[key]

            # only have to add what isn't already there
            set_n = run['set_n']
            max_speed = run['max_speed']
            run["n"] = rep
            run['run_name'] = f"{run['planner_type']}_{run['algorithm']}_{run['state_vector']}_{run['map_name']}_{run['id_name']}_{max_speed}_{set_n}_{rep}"
            run['path'] = f"{test_name}/"

            run_list.append(Namespace(**run))

    return run_list


@njit(cache=True)
def calculate_speed(delta, f_s=0.8, max_v=7):
    b = 0.523
    g = 9.81
    l_d = 0.329

    if abs(delta) < 0.03:
        return max_v
    if abs(delta) > 0.4:
        return 0

    V = f_s * np.sqrt(b*g*l_d/np.tan(abs(delta)))

    V = min(V, max_v)

    return V

def save_csv_array(data, filename):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def moving_average(data, period):
    return np.convolve(data, np.ones(period), 'same') / period

def true_moving_average(data, period):
    if len(data) < period:
        return np.zeros_like(data)
    ret = np.convolve(data, np.ones(period), 'same') / period
    # t_end = np.convolve(data, np.ones(period), 'valid') / (period)
    # t_end = t_end[-1] # last valid value
    for i in range(period): # start
        t = np.convolve(data, np.ones(i+2), 'valid') / (i+2)
        ret[i] = t[0]
    for i in range(period):
        length = int(round((i + period)/2))
        t = np.convolve(data, np.ones(length), 'valid') / length
        ret[-i-1] = t[-1]
    return ret

def save_run_config(run_dict, path):
    path = path +  f"/TrainingRunDict_record.yaml"
    with open(path, 'w') as file:
        yaml.dump(run_dict, file)


    
def profile_and_save(function):
    with cProfile.Profile(builtins=False) as pr:
        function()
        
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