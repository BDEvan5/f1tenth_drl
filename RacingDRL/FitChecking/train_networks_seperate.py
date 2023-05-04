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
        # loss = l_steering + l_speed
        
        return mu, l_steering, l_speed
    
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
    
    train_iterations = 401
    # train_iterations = 301
    train_losses, test_losses = [], []
    test_loss_steering_array, test_loss_speed_array = [], []
    
    for i in range(train_iterations):
        network.eval()
        test_pred_y, test_loss_steering, test_loss_speed = network.forward_loss(test_x, test_y)
        network.train()
        
        pred_y, loss_steering, loss_speed = network.forward_loss(train_x, train_y)
        loss = loss_steering + loss_speed
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item() ** 0.5
        loss_speed = test_loss_speed.item() ** 0.5
        loss_steering = test_loss_steering.item() ** 0.5
        test_losses.append(loss_speed + loss_steering)
        test_loss_speed_array.append(loss_speed)
        test_loss_steering_array.append(loss_steering)
        train_losses.append(train_loss)
        
        if i % 50 == 0:
            print(f"{i}: TrainLoss: {train_loss}")
            # l_steer, l_speed = network.separate_losses(test_x, test_y)
            print(f"SpeedLoss: {test_loss_speed.item()**0.5} --> SteerLoss: {test_loss_steering.item()**0.5}")
         
    # plot the speed and steering losses
    plt.figure()
    plt.plot(test_loss_speed_array, label="Speed")
    plt.plot(test_loss_steering_array, label="Steering")
    plt.legend()
    plt.title(f"Test Losses: {name}_{seed}")
    plt.ylim(0, 0.8)
    
    plt.savefig(folder + f"LossResultsSeperate/{name}_{seed}.svg")
    # plt.show()
    
         
    if not os.path.exists(folder + "Models/"): os.mkdir(folder + "Models/")   
    torch.save(network, folder + f"Models/{name}_{seed}.pt")
    
    return train_losses, test_loss_speed_array, test_loss_steering_array
    
    
def run_seeded_test(folder, key, seeds):
    train_losses, test_losses_speed, test_losses_steering = [], [], []    
    for i in range(len(seeds)):
        np.random.seed(seeds[i])
        train_loss, test_loss_speed, test_loss_steering = train_networks(folder, key, seeds[i])
        
        train_losses.append(train_loss)
        test_losses_speed.append(test_loss_speed)
        test_losses_steering.append(test_loss_steering)
    
    train_losses = np.array(train_losses)
    test_losses_steering = np.array(test_losses_steering)
    test_losses_speed = np.array(test_losses_speed)
    
    return train_losses, test_losses_steering, test_losses_speed
        
    
def run_experiment(folder, name_keys, experiment_name, n_seeds=5):
    save_path = folder + "LossResultsSeperate/"
    if not os.path.exists(save_path): os.mkdir(save_path)
    spacing = 18
    with open(folder + f"LossResultsSeperate/{experiment_name}_LossResultsSeperate.txt", "w") as f:
        f.write(f"Name".ljust(30))
        f.write(f"TrainLoss mean".rjust(spacing))
        f.write(f"TrainLoss std".rjust(spacing))
        f.write(f"TestSteer mean".rjust(spacing))
        f.write(f"TestSteer std".rjust(spacing))
        f.write(f"TestSpeed mean".rjust(spacing))
        f.write(f"TestSpeed std".rjust(spacing))
        f.write(f"\n")
        
    seeds = np.arange(n_seeds)
    for key in name_keys:
        train_losses, test_losses_steering, test_losses_speed = run_seeded_test(folder, key, seeds)
        
        # Add some individual plotting here....
        np.save(folder + f"LossResultsSeperate/{experiment_name}_{key}_train_losses.npy", train_losses)
        np.save(folder + f"LossResultsSeperate/{experiment_name}_{key}_test_losses_speed.npy", test_losses_speed)
        np.save(folder + f"LossResultsSeperate/{experiment_name}_{key}_test_losses_steering.npy", test_losses_steering)
        
        with open(folder + f"LossResultsSeperate/{experiment_name}_LossResultsSeperate.txt", "a") as f:
            f.write(f"{key},".ljust(30) )
            f.write(f"{np.mean(train_losses[:, -1]):.5f},".rjust(spacing))
            f.write(f"{np.std(train_losses[:, -1]):.5f},".rjust(spacing))
            
            f.write(f"{np.mean(test_losses_steering[:, -1]):.5f},".rjust(spacing))
            f.write(f"{np.std(test_losses_steering[:, -1]):.5f},".rjust(spacing))
            
            f.write(f"{np.mean(test_losses_speed[:, -1]):.5f},".rjust(spacing))
            f.write(f"{np.std(test_losses_speed[:, -1]):.5f},".rjust(spacing))
            
            f.write(f"\n")
            
        
        
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
    run_experiment(folder, name_keys, name, 5)
    
def run_planningAblation_test():
    set_n = 3
    name = "fullPlanning_ablation"
    folder = f"NetworkFitting/{name}_{set_n}/"
    name_keys = ["fullPlanning_full", "fullPlanning_rmMotion", "fullPlanning_rmLidar", "fullPlanning_rmWaypoints"]
    run_experiment(folder, name_keys, name, 5)
    
        
def run_comparison_test():
    set_n = 3
    name = "comparison"
    folder = f"NetworkFitting/{name}_{set_n}/"
    name_keys = ["fullPlanning", "trajectoryTrack", "endToEnd", "endToEnd_Single"]
    run_experiment(folder, name_keys, name)
    
    
    
     
if __name__ == "__main__":
    # run_nBeams_test()
    # run_endStacking_test()
    # run_planningAblation_test()
    
    run_comparison_test()
    
    