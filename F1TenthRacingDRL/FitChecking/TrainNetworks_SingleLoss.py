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
        loss = l_steering + l_speed
        
        return mu, loss
    
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
    trains = []
    tests = []
    for i in range(10):
        x, y = make_minibatch(train_x, train_y, batch_size=BATCH_SIZE)
        _, train_loss = model.forward_loss(x, y)
        x, y = make_minibatch(test_x, test_y, batch_size=BATCH_SIZE)
        _, test_loss = model.forward_loss(x, y)
        trains.append(train_loss.item()** 0.5)
        tests.append(test_loss.item()** 0.5)
        
    trains = np.mean(trains)
    tests = np.mean(tests)        
        
    model.train()
    
    return trains, tests
    
def make_minibatch(x, y, batch_size):
    inds = np.random.choice(x.shape[0], size=batch_size, replace=False)
    x = x[inds]
    y = y[inds]
    
    return x, y
    
def train_networks(folder, name, seed):
    train_x, train_y, test_x, test_y = load_data(folder, name)
    
    network = StdNetworkTwo(train_x.shape[1], train_y.shape[1])
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    
    train_iterations = 2000
    train_losses, test_losses = [], []
    
    for i in range(train_iterations):
        x, y = make_minibatch(train_x, train_y, batch_size=BATCH_SIZE)
        pred_y, loss = network.forward_loss(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            train_loss, test_loss = estimate_losses(train_x, train_y, test_x, test_y, network)
            test_losses.append(test_loss)
            train_losses.append(train_loss)
            print(f"{i}: TrainLoss: {train_loss} --> TestLoss: {test_loss}")
         
    if not os.path.exists(folder + "Models/"): os.mkdir(folder + "Models/")   
    torch.save(network, folder + f"Models/{name}_{seed}.pt")
    
    return train_losses, test_losses
    
    
def run_seeded_test(folder, key, seeds):
    train_losses, test_losses = [], []    
    for i in range(len(seeds)):
        np.random.seed(seeds[i])
        train_loss, test_loss = train_networks(folder, key, seeds[i])
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        plt.figure(1)
        plt.clf()
        xs = np.arange(len(train_loss)) * 50
        plt.plot(xs, train_loss, label="Train loss")
        plt.plot(xs, test_loss, label="Test loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(folder + f"LossResults/{key}_{i}_LossResults.svg", pad_inches=0, bbox_inches='tight')
        
        plt.ylim(0, 0.1)
        plt.savefig(folder + f"LossResults/{key}_{i}_LossResultsZoom.svg", pad_inches=0, bbox_inches='tight')
    
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    
    return train_losses, test_losses
        
    
def run_experiment(folder, name_keys, experiment_name, n_seeds=3):
    save_path = folder + "LossResults/"
    if not os.path.exists(save_path): os.mkdir(save_path)
    spacing = 30
    with open(folder + f"LossResults/{experiment_name}_LossResults.txt", "w") as f:
        f.write(f"Name,".ljust(spacing))
        f.write(f"TrainLoss mean, TrainLoss std ,   TestLoss mean, TestLoss std \n")
        
    seeds = np.arange(n_seeds)
    for key in name_keys:
        train_losses, test_losses = run_seeded_test(folder, key, seeds)
        
        np.save(folder + f"LossResults/{experiment_name}_{key}_train_losses.npy", train_losses)
        np.save(folder + f"LossResults/{experiment_name}_{key}_test_losses.npy", test_losses)
        
        with open(folder + f"LossResults/{experiment_name}_LossResults.txt", "a") as f:
            f.write(f"{key},".ljust(spacing))
            f.write(f"{np.mean(train_losses[:, -1]):.5f},     {np.std(train_losses[:, -1]):.5f},       {np.mean(test_losses[:, -1]):.5f},       {np.std(test_losses[:, -1]):.5f} \n")
        
        
        
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
    name_keys = ["endToEnd_Single", "endToEnd_Double", "endToEnd_Triple", "endToEnd_Speed"]
    run_experiment(folder, name_keys, name)
    
        
            
def run_trajectoryWaypoints_test():
    set_n = 3
    name = "trajectoryTrack_nWaypoints"
    folder = f"TuningData/{name}_{set_n}/"
    # inds = [0, 1, 2, 5, 10]
    inds = [0, 1, 2, 4, 6, 8, 10, 12, 15, 20]
    name_keys = [f"trajectoryTrack_{i}" for i in inds]
    run_experiment(folder, name_keys, name, 3)
               
               
def run_planningAblation_test():
    set_n = 3
    name = "fullPlanning_ablation"
    folder = f"TuningData/{name}_{set_n}/"
    name_keys = ["fullPlanning_full", "fullPlanning_rmMotion", "fullPlanning_rmLidar", "fullPlanning_rmWaypoints", "fullPlanning_Motion"]
    run_experiment(folder, name_keys, name)
    
    
def run_comparison_test():
    set_n = 1
    name = "comparison"
    folder = f"TuningData/{name}_{set_n}/"
    name_keys = ["fullPlanning", "trajectoryTrack", "endToEnd"]
    run_experiment(folder, name_keys, name, 10)
    
    
        
def run_TuningNN_test():
    set_n = 1
    name = "TuningNN"
    folder = f"TuningData/{name}_{set_n}/"
    name_keys = ["fullPlanning"]
    run_experiment(folder, name_keys, name, 10)
    
    
    
     
if __name__ == "__main__":
    # run_nBeams_test()
    # run_endStacking_test()
    
    # run_trajectoryWaypoints_test()
    # run_planningAblation_test()
    
    # run_comparison_test()
    run_TuningNN_test()
    