import torch
import torch.nn as nn
import torch.nn.functional as F
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

import numpy as np
import os
from matplotlib import pyplot as plt



BATCH_SIZE = 100

class StdNetworkTwo(nn.Module):
    def __init__(self, state_dim, act_dim, layer_sz):
        super(StdNetworkTwo, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, layer_sz)
        self.fc2 = nn.Linear(layer_sz, layer_sz)
        self.fc_mu = nn.Linear(layer_sz, act_dim)
        
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
    
def train_networks(folder, name, seed, network_sz):
    train_x, train_y, test_x, test_y = load_data(folder, name)
    
    network = StdNetworkTwo(train_x.shape[1], train_y.shape[1], network_sz)
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
    torch.save(network, folder + f"Models/{name}_{network_sz}_{seed}.pt")
    
    return train_losses, test_losses
    
    
def run_seeded_test(folder, key, seeds, network_sz):
    train_losses, test_losses = [], []    
    for i in range(len(seeds)):
        np.random.seed(seeds[i])
        train_loss, test_loss = train_networks(folder, key, seeds[i], network_sz)
        
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
        
    
def tune_neural_network_size(n_seeds=10):
    set_n = 1
    experiment_name = "TuningNN"
    folder = f"TuningData/{experiment_name}_{set_n}/"
    # name_key = "fullPlanning"
    # name_key = "endToEnd"
    name_key = "trajectoryTrack"
    save_path = folder + "LossResultsT/"
    # save_path = folder + "LossResultsF/"
    # save_path = folder + "LossResultsE/"

    if not os.path.exists(save_path): os.mkdir(save_path)
    spacing = 30
    # with open(f"{save_path}{experiment_name}_LossResults.txt", "w") as f:
    #     f.write(f"Name,".ljust(spacing))
    #     f.write(f"TrainLoss mean, TrainLoss std ,   TestLoss mean, TestLoss std \n")
        
    network_sizes = np.array([20, 50, 80, 100, 150, 200, 250, 300])
    # network_sizes = np.array([10, 15, 20, 50, 80, 100, 150, 200, 250, 300])
    # network_sizes = np.array([10, 15, 30])
    # network_sizes = np.array([10, 15])
    seeds = np.arange(n_seeds)
    for network_sz in network_sizes:
        train_losses, test_losses = run_seeded_test(folder, name_key, seeds, network_sz)
        
        np.save(f"{save_path}{experiment_name}_{network_sz}_train_losses.npy", train_losses)
        np.save(f"{save_path}{experiment_name}_{network_sz}_test_losses.npy", test_losses)
        
        # with open(f"{save_path}{experiment_name}_LossResults.txt", "a") as f:
        #     f.write(f"{network_sz},".ljust(spacing))
        #     f.write(f"{np.mean(train_losses[:, -1]):.5f},     {np.std(train_losses[:, -1]):.5f},       {np.mean(test_losses[:, -1]):.5f},       {np.std(test_losses[:, -1]):.5f} \n")
        
        with open(f"{save_path}{experiment_name}_LossResults_q.txt", "a") as f:
            f.write(f"{network_sz},".ljust(spacing))
            f.write(f"{np.mean(train_losses[:, -1]):.5f},     {np.percentile(train_losses[:, -1], 25):.5f},         {np.percentile(train_losses[:, -1], 75):.5f},       {np.mean(test_losses[:, -1]):.5f},       {np.percentile(test_losses[:, -1], 25):.5f},         {np.percentile(test_losses[:, -1], 75):.5f} \n")
        
        
        
     
if __name__ == "__main__":
    tune_neural_network_size()
    