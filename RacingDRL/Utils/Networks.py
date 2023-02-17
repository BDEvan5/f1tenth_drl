
import torch
import torch.nn as nn
import torch.nn.functional as F

NN_LAYER_1 = 400
NN_LAYER_2 = 300

class DoublePolicyNet(nn.Module):
    def __init__(self, state_dim, act_dim, action_scale):
        super(DoublePolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_mu = nn.Linear(NN_LAYER_2, act_dim)

        self.action_scale = action_scale

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * self.action_scale
        return mu

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6

class PolicyNetworkSAC(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PolicyNetworkSAC, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, NN_LAYER_1)
        self.linear2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.mean_linear = nn.Linear(NN_LAYER_2, num_actions)
        self.log_std_linear = nn.Linear(NN_LAYER_2, num_actions)
    
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(0, 1) # assumes actions have been normalized to (0,1)
        
        z = mean + std * normal.sample().requires_grad_()
        action = torch.tanh(z)
        log_prob = torch.distributions.Normal(mean, std).log_prob(z) - torch.log(1 - action * action + EPSILON) 
            
        return action, log_prob
   

class DoubleQNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DoubleQNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + act_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_out = nn.Linear(NN_LAYER_2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x2 = F.relu(self.fc1(x))
        x3 = F.relu(self.fc2(x2))
        q = self.fc_out(x3)
        return q

class QNetworkDQN(nn.Module):
    def __init__(self, obs_space, action_space):
        super(QNetworkDQN, self).__init__()
        self.fc1 = nn.Linear(obs_space, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc3 = nn.Linear(NN_LAYER_2, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SingleActor(nn.Module):
    def __init__(self, obs_space, action_space):
        super(SingleActor, self).__init__()
        self.fc1 = nn.Linear(obs_space, NN_LAYER_1)
        self.fc_pi = nn.Linear(NN_LAYER_1, action_space)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        probs = F.softmax(x, dim=softmax_dim)
        
        return probs
        
    
class SingleVNet(nn.Module):
    def __init__(self, obs_space):
        super(SingleVNet, self).__init__()
        self.fc1 = nn.Linear(obs_space, NN_LAYER_1)
        self.fc_v  = nn.Linear(NN_LAYER_1, 1)

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    

if __name__ == "__main__":
    print("Hello World!")
    
    