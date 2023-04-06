import torch
import torch.nn as nn
import torch.nn.functional as F
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

import numpy as np
from matplotlib import pyplot as plt


NN_LAYER_1 = 100
NN_LAYER_2 = 100


set_n = 3
folder = f"Data/PurePursuitDataGen_{set_n}/"



class StdNetworkTwo(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(StdNetworkTwo, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_mu = nn.Linear(NN_LAYER_2, act_dim)
        
        # self.dropout = nn.Dropout(p=0.2)

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
    
    
def load_data(name):
    
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
    
def train_networks(name):
    train_x, train_y, test_x, test_y = load_data(name)
    
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
            
    torch.save(network, folder + f"Models/{name}.pt")
        
    
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
    
def run_seeded_test(key, seeds):
    train_losses, test_losses = [], []    
    for i in range(len(seeds)):
        np.random.seed(seeds[i])
        train_loss, test_loss = train_networks(key)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    
    return train_losses, test_losses
        
    
def run_endToEnd_nBeams():
    name = "endToEndState"
    with open(folder + f"LossImgs/{name}_LossResults.txt", "w") as f:
        f.write(f"Name        TrainLoss mean, TrainLoss std ,   TestLoss mean, TestLoss std \n")
        
    seeds = np.arange(5)
    # name_keys = ["endToEnd_5", "endToEnd_10", "endToEnd_20", "endToEnd_30"]
    name_keys = ["endToEnd_5", "endToEnd_10", "endToEnd_12","endToEnd_15", "endToEnd_20", "endToEnd_30", "endToEnd_60"]
    for key in name_keys:
        train_losses, test_losses = run_seeded_test(key, seeds)
        
        # Add some individual plotting here....
        
        with open(folder + f"LossImgs/{name}_LossResults.txt", "a") as f:
            f.write(f"{key},  {np.mean(train_losses[:, -1]):.5f},     {np.std(train_losses[:, -1]):.5f},       {np.mean(test_losses[:, -1]):.5f},       {np.std(test_losses[:, -1]):.5f} \n")
        
    
        

    # plt.figure(figsize=(10, 6))
    # for i in range(len(name_keys)):
    #     plt.plot(train_losses[i], label=name_keys[i], linewidth=2)
        
    # plt.title("Training losses")
    # plt.legend()
    # plt.grid(True)
    # plt.ylabel("RMSE")
    # plt.xlabel("Iteration")
    # plt.tight_layout()
    
    # plt.savefig(folder + f"LossImgs/TrainLosses.svg")
    
    # plt.clf()
    # for i in range(len(name_keys)):
    #     plt.plot(test_losses[i], label=name_keys[i], linewidth=2)
        
    # plt.title("Test losses")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.ylabel("RMSE")
    # plt.xlabel("Iteration")
    
    # plt.savefig(folder + f"LossImgs/TestLosses.svg")
    

    
    
    
    
if __name__ == "__main__":
    run_endToEnd_nBeams()
    from RacingDRL.FitChecking.plot_n_beams import plot_n_beams
    # plot_n_beams()

