import torch
import torch.nn as nn
import torch.nn.functional as F
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

import numpy as np
import os
from matplotlib import pyplot as plt


NN_LAYER_1 = 100
NN_LAYER_2 = 100

BATCH_SIZE = 100

class StdNetworkTwo(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(StdNetworkTwo, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_mu = nn.Linear(NN_LAYER_2, act_dim)
        
    def forward_loss(self, x, targets):
        mu = self.forward(x)
        l_steering = F.mse_loss(mu[:, 0], targets[:, 0])
        l_speed = F.mse_loss(mu[:, 1], targets[:, 1])
        
        return mu, l_steering, l_speed
    
    def separate_losses(self, x, targets):
        mu = self.forward(x)
        l_steering = F.mse_loss(mu[:, 0], targets[:, 0])
        l_speed = F.mse_loss(mu[:, 1], targets[:, 1])
        
        return l_steering, l_speed
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) 
        
        return mu
    
    
def load_data(folder, name):
    
    states = np.load(folder + f"DataSets/PurePursuit_{name}_states.npy")
    actions = np.load(folder + f"DataSets/PurePursuit_actions.npy")
    
    # test_size = 200
    test_size = int(0.1*states.shape[0])
    test_inds = np.random.choice(states.shape[0], size=test_size, replace=False)
    
    test_x = states[test_inds]
    test_y = actions[test_inds]
    
    train_x = states[~np.isin(np.arange(states.shape[0]), test_inds)]
    train_y = actions[~np.isin(np.arange(states.shape[0]), test_inds)]
    
    test_x = torch.FloatTensor(test_x)
    test_y = torch.FloatTensor(test_y)
    train_x = torch.FloatTensor(train_x)
    train_y = torch.FloatTensor(train_y)
    
    print(f"Train: {train_x.shape} --> Test: {test_x.shape}")
    
    return train_x, train_y, test_x, test_y
    

def estimate_losses(train_x, train_y, test_x, test_y, model):
    model.eval()
    train_steers, train_speeds = [], []
    test_steers, test_speeds = [], []
    for i in range(10):
        x, y = make_minibatch(train_x, train_y, batch_size=BATCH_SIZE)
        train_steer, train_speed = model.separate_losses(x, y)
        train_steers.append(train_steer.item()** 0.5)
        train_speeds.append(train_speed.item()** 0.5)
        
        x, y = make_minibatch(test_x, test_y, batch_size=BATCH_SIZE)
        test_steer, test_speed = model.separate_losses(x, y)
        test_steers.append(test_steer.item()** 0.5)
        test_speeds.append(test_speed.item()** 0.5)
        
    train_steer = np.mean(train_steers)
    train_speed = np.mean(train_speeds)        
    test_steer = np.mean(test_steers)        
    test_speed = np.mean(test_speeds)        
        
    model.train()
    
    return train_steer, train_speed, test_steer, test_speed
    
def make_minibatch(x, y, batch_size):
    inds = np.random.choice(x.shape[0], size=batch_size, replace=False)
    x = x[inds]
    y = y[inds]
    
    return x, y
    
def train_networks(folder, name, seed):
    train_x, train_y, test_x, test_y = load_data(folder, name)
    
    network = StdNetworkTwo(train_x.shape[1], train_y.shape[1])
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    
    train_iterations = 2001
    test_loss_steering_array, test_loss_speed_array = [], []
    train_loss_steering_array, train_loss_speed_array = [], []
    
    for i in range(train_iterations):
        x, y = make_minibatch(train_x, train_y, batch_size=BATCH_SIZE)
        pred_y, loss_steering, loss_speed = network.forward_loss(x, y)
        loss = loss_steering + loss_speed
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            train_loss_steer, train_loss_speed, test_loss_steer, test_loss_speed = estimate_losses(train_x, train_y, test_x, test_y, network)
            
            test_loss_speed_array.append(test_loss_speed)
            test_loss_steering_array.append(test_loss_steer)
            train_loss_steering_array.append(train_loss_steer)
            train_loss_speed_array.append(train_loss_speed)
            
            print(f"{i}: SpeedLoss: {test_loss_speed} --> SteerLoss: {test_loss_steer}")
         
    # plot the speed and steering losses
    plt.figure()
    plt.plot(test_loss_speed_array, label="Speed")
    plt.plot(test_loss_steering_array, label="Steering")
    plt.legend()
    plt.title(f"Test Losses: {name}_{seed}")
    plt.ylim(0, 0.12)
    
    plt.savefig(folder + f"LossResultsSeperate/{name}_{seed}.svg")
    # plt.show()
    
         
    if not os.path.exists(folder + "Models/"): os.mkdir(folder + "Models/")   
    torch.save(network, folder + f"Models/{name}_{seed}.pt")
    
    return train_loss_steering_array, train_loss_speed_array, test_loss_steering_array, test_loss_speed_array
    
    
def run_seeded_test(folder, key, seeds):
    train_losses_speed, train_losses_steering = [], []  
    test_losses_speed, test_losses_steering = [], []    
    for i in range(len(seeds)):
        np.random.seed(seeds[i])
        train_loss_steering, train_loss_speed, test_loss_steering, test_loss_speed = train_networks(folder, key, seeds[i])
        
        train_losses_speed.append(train_loss_speed)
        train_losses_steering.append(train_loss_steering)
        test_losses_speed.append(test_loss_speed)
        test_losses_steering.append(test_loss_steering)
    
    train_losses_steering = np.array(train_losses_steering)
    train_losses_speed = np.array(train_losses_speed)
    test_losses_steering = np.array(test_losses_steering)
    test_losses_speed = np.array(test_losses_speed)
    
    return train_losses_steering, train_losses_speed, test_losses_steering, test_losses_speed
        
    
def run_experiment(folder, name_keys, experiment_name, n_seeds=5):
    save_path = folder + "LossResultsSeperate/"
    if not os.path.exists(save_path): os.mkdir(save_path)
    spacing = 18
    with open(folder + f"LossResultsSeperate/{experiment_name}_LossResultsSeperate.txt", "w") as f:
        f.write(f"Name".ljust(30))
        f.write(f"TrainSteer mean".rjust(spacing))
        f.write(f"TrainSteer std".rjust(spacing))
        f.write(f"TrainSpeed mean".rjust(spacing))
        f.write(f"TrainSpeed std".rjust(spacing))
        f.write(f"TestSteer mean".rjust(spacing))
        f.write(f"TestSteer std".rjust(spacing))
        f.write(f"TestSpeed mean".rjust(spacing))
        f.write(f"TestSpeed std".rjust(spacing))
        f.write(f"\n")
        
    seeds = np.arange(n_seeds)
    for key in name_keys:
        train_losses_steering, train_losses_speed, test_losses_steering, test_losses_speed = run_seeded_test(folder, key, seeds)
        
        # Add some individual plotting here....
        np.save(folder + f"LossResultsSeperate/{experiment_name}_{key}_train_losses_speed.npy", train_losses_speed)
        np.save(folder + f"LossResultsSeperate/{experiment_name}_{key}_train_losses_steering.npy", train_losses_steering)
        np.save(folder + f"LossResultsSeperate/{experiment_name}_{key}_test_losses_speed.npy", test_losses_speed)
        np.save(folder + f"LossResultsSeperate/{experiment_name}_{key}_test_losses_steering.npy", test_losses_steering)
        
        with open(folder + f"LossResultsSeperate/{experiment_name}_LossResultsSeperate.txt", "a") as f:
            f.write(f"{key},".ljust(30) )
            
            f.write(f"{np.mean(train_losses_steering[:, -1]):.5f},".rjust(spacing))
            f.write(f"{np.std(train_losses_steering[:, -1]):.5f},".rjust(spacing))
            
            f.write(f"{np.mean(train_losses_speed[:, -1]):.5f},".rjust(spacing))
            f.write(f"{np.std(train_losses_speed[:, -1]):.5f},".rjust(spacing))
                        
            f.write(f"{np.mean(test_losses_steering[:, -1]):.5f},".rjust(spacing))
            f.write(f"{np.std(test_losses_steering[:, -1]):.5f},".rjust(spacing))
            
            f.write(f"{np.mean(test_losses_speed[:, -1]):.5f},".rjust(spacing))
            f.write(f"{np.std(test_losses_speed[:, -1]):.5f},".rjust(spacing))
            
            f.write(f"\n")
            
        
        
def run_nBeams_test():
    set_n = 3
    name = "EndToEnd_nBeams"
    folder = f"TuningData/{name}_{set_n}/"
    name_keys = ["endToEnd_5", "endToEnd_10", "endToEnd_12","endToEnd_15", "endToEnd_20", "endToEnd_30", "endToEnd_60"]
    run_experiment(folder, name_keys, name)
    
    
            
def run_endStacking_test():
    set_n = 3
    name = "EndToEnd_stacking"
    folder = f"TuningData/{name}_{set_n}/"
    name_keys = ["endToEnd_Single", "endToEnd_Double", "endToEnd_Triple", "endToEnd_Speed", "endToEnd_DoubleSpeed"]
    run_experiment(folder, name_keys, name, 3)
    
def run_planningAblation_test():
    set_n = 3
    name = "fullPlanning_ablation"
    folder = f"TuningData/{name}_{set_n}/"
    name_keys = ["fullPlanning_full", "fullPlanning_rmMotion", "fullPlanning_rmLidar", "fullPlanning_rmWaypoints", "fullPlanning_Motion"]
    run_experiment(folder, name_keys, name, 3)
    
        
def run_comparison_test():
    set_n = 3
    name = "comparison"
    folder = f"TuningData/{name}_{set_n}/"
    name_keys = ["fullPlanning", "trajectoryTrack", "endToEnd"]
    # name_keys = ["fullPlanning", "trajectoryTrack", "endToEnd", "endToEnd_Single"]
    run_experiment(folder, name_keys, name, 10)
    
    
    
     
if __name__ == "__main__":
    # run_nBeams_test()
    # run_endStacking_test()
    # run_planningAblation_test()
    
    run_comparison_test()
    
    