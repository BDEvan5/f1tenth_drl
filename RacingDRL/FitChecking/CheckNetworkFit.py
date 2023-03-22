import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt


NN_LAYER_1 = 100
NN_LAYER_2 = 100

folder = "Data/PurePursuitDataGen_2/"


class StdNetworkTwo(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(StdNetworkTwo, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_mu = nn.Linear(NN_LAYER_2, act_dim)

    def forward_loss(self, x, targets):
        mu = self.forward(x)
        loss = F.mse_loss(mu, targets)
        
        return mu, loss
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) 
        
        return mu
    
def load_data(name):
    
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
    
def train_networks(name):
    train_x, train_y, test_x, test_y = load_data(name)
    
    network = StdNetworkTwo(train_x.shape[1], train_y.shape[1])
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    
    train_iterations = 300
    train_losses, test_losses = [], []
    
    for i in range(train_iterations):
        test_pred_y, test_loss = network.forward_loss(test_x, test_y)
        
        test_losses.append(test_loss.item())
        
        pred_y, loss = network.forward_loss(train_x, train_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        print(f"{i}: TrainLoss: {loss.item()} --> TestLoss: {test_loss.item()}")
        
    plot_losses(train_losses, test_losses, name)
    
    return train_losses, test_losses
    
        
def plot_losses(train, test, name="endToEnd"):
    plt.figure(1)
    plt.figure(figsize=(10,6))
    plt.plot(train, label="Train")
    plt.plot(test, label="Test")
    
    plt.title("Losses")
    plt.xlabel("Iteration")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(folder + f"LossImgs/Losses_{name}.svg")
        
def train_all_networks():
    train_losses, test_losses = [], []    
    
    name_keys = ["endToEnd", "Game", "trajFollow"]
    for key in name_keys:
        train_loss, test_loss = train_networks(key)
    
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(train_losses[i], label=name_keys[i])
        
    plt.title("Training losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(folder + f"LossImgs/TrainLosses.svg")
    
    plt.clf()
    for i in range(3):
        plt.plot(test_losses[i], label=name_keys[i])
        
    plt.title("Test losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(folder + f"LossImgs/TestLosses.svg")
    
train_all_networks()