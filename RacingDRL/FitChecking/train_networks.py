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
        # l_steering = F.mse_loss(mu[:, 0], targets[:, 0], reduction='sum')
        # l_speed = F.mse_loss(mu[:, 1], targets[:, 1], reduction='sum')
        loss = l_steering + l_speed
        
        return mu, loss
    
    def separate_losses(self, x, targets):
        mu = self.forward(x)
        l_steering = F.mse_loss(mu[:, 0], targets[:, 0])
        l_speed = F.mse_loss(mu[:, 1], targets[:, 1])
        loss = l_steering + l_speed
        
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
    
def train_networks(folder, name, seed):
    train_x, train_y, test_x, test_y = load_data(folder, name)
    
    network = StdNetworkTwo(train_x.shape[1], train_y.shape[1])
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    
    train_iterations = 301
    train_losses, test_losses = [], []
    
    for i in range(train_iterations):
        network.eval()
        test_pred_y, test_loss = network.forward_loss(test_x, test_y)
        network.train()
        
        pred_y, loss = network.forward_loss(train_x, train_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        test_loss = test_loss.item() ** 0.5
        train_loss = loss.item() ** 0.5
        test_losses.append(test_loss)
        train_losses.append(train_loss)
        
        if i % 50 == 0:
            print(f"{i}: TrainLoss: {train_loss} --> TestLoss: {test_loss}")
            l_steer, l_speed = network.separate_losses(test_x, test_y)
            print(f"SteerLoss: {l_steer**0.5} --> SpeedLoss: {l_speed**0.5}")
         
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
    
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    
    return train_losses, test_losses
        
    
def run_experiment(folder, name_keys, experiment_name):
    save_path = folder + "LossResults/"
    if not os.path.exists(save_path): os.mkdir(save_path)
    spacing = 30
    with open(folder + f"LossResults/{experiment_name}_LossResults.txt", "w") as f:
        f.write(f"Name,".ljust(spacing))
        f.write(f"TrainLoss mean, TrainLoss std ,   TestLoss mean, TestLoss std \n")
        
    seeds = np.arange(5)
    for key in name_keys:
        train_losses, test_losses = run_seeded_test(folder, key, seeds)
        
        # Add some individual plotting here....
        np.save(folder + f"LossResults/{experiment_name}_{key}_train_losses.npy", train_losses)
        np.save(folder + f"LossResults/{experiment_name}_{key}_test_losses.npy", test_losses)
        
        with open(folder + f"LossResults/{experiment_name}_LossResults.txt", "a") as f:
            f.write(f"{key},".ljust(spacing))
            f.write(f"{np.mean(train_losses[:, -1]):.5f},     {np.std(train_losses[:, -1]):.5f},       {np.mean(test_losses[:, -1]):.5f},       {np.std(test_losses[:, -1]):.5f} \n")
        
        
        
def run_nBeams_test():
    set_n = 3
    name = "EndToEnd_nBeams"
    folder = f"NetworkFitting/{name}_{set_n}/"
    name_keys = ["endToEnd_5", "endToEnd_10", "endToEnd_12","endToEnd_15", "endToEnd_20", "endToEnd_30", "endToEnd_60"]
    run_experiment(folder, name_keys, name)
    
    
            
def run_endStacking_test():
    set_n = 3
    name = "EndToEnd_stacking"
    folder = f"NetworkFitting/{name}_{set_n}/"
    name_keys = ["endToEnd_Single", "endToEnd_Double", "endToEnd_Triple", "endToEnd_Speed"]
    run_experiment(folder, name_keys, name)
    
        
            
def run_trajectoryWaypoints_test():
    set_n = 3
    name = "trajectoryTrack_nWaypoints"
    folder = f"NetworkFitting/{name}_{set_n}/"
    # inds = [0, 1, 2, 5, 10]
    inds = [0, 1, 2, 4, 6, 8, 10, 12, 15, 20]
    name_keys = [f"trajectoryTrack_{i}" for i in inds]
    run_experiment(folder, name_keys, name)
               
def run_planningAblation_test():
    set_n = 3
    name = "fullPlanning_ablation"
    folder = f"NetworkFitting/{name}_{set_n}/"
    name_keys = ["fullPlanning_full", "fullPlanning_rmMotion", "fullPlanning_rmLidar", "fullPlanning_rmWaypoints", "fullPlanning_Motion"]
    run_experiment(folder, name_keys, name)
    
    
    
     
if __name__ == "__main__":
    # run_nBeams_test()
    # run_endStacking_test()
    
    # run_trajectoryWaypoints_test()
    run_planningAblation_test()
    