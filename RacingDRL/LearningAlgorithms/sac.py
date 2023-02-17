import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import soft_update

torch.autograd.set_detect_anomaly(True)

SEED = 0
TAU = 1e-2
GAMMA = 0.99
BATCH_SIZE = 100
LR = 1e-3

   
from Components.ReplayBuffers import OffPolicyBuffer
from Components.Networks import PolicyNetworkSAC, DoubleQNet


class SAC(object):
    def __init__(self, state_dim, action_dim):
        self.replay_buffer = OffPolicyBuffer(state_dim, action_dim)

        self.soft_q_net1 = DoubleQNet(state_dim, action_dim)
        self.soft_q_net2 = DoubleQNet(state_dim, action_dim)
        self.target_soft_q_net1 = DoubleQNet(state_dim, action_dim)
        self.target_soft_q_net2 = DoubleQNet(state_dim, action_dim)
        self.target_soft_q_net1.load_state_dict(self.soft_q_net1.state_dict())
        self.target_soft_q_net2.load_state_dict(self.soft_q_net2.state_dict())
        
        self.policy_net = PolicyNetworkSAC(state_dim, action_dim)
        
        self.soft_q_criterion = nn.MSELoss()
        self.q_optimiser = optim.Adam(list(self.soft_q_net1.parameters()) + list(self.soft_q_net2.parameters()), lr=LR)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        
        self.target_entropy = -np.prod(action_dim).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR)
        
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ =  self.policy_net(state)
        return action.detach()[0].numpy()
               
    def train(self, iterations=2):
        for _ in range(0, iterations):
            state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
            alpha = self.log_alpha.exp()
            
            self.update_policy(state, alpha)
            self.update_Q(state, action, next_state, reward, done, alpha)
            self.update_alpha(state)
            
            soft_update(self.soft_q_net1, self.target_soft_q_net1, TAU)
            soft_update(self.soft_q_net2, self.target_soft_q_net2, TAU)
        
    def update_policy(self, state, alpha):
        new_actions, log_pi = self.policy_net(state)

        q1 = self.soft_q_net1(state, new_actions)
        q2 = self.soft_q_net2(state, new_actions)
        q_new_actions = torch.min(q1, q2)
        
        alpha = self.log_alpha.exp()
        policy_loss = (alpha*log_pi - q_new_actions).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
    def update_Q(self, state, action, next_state, reward, done, alpha):
            current_q1 = self.soft_q_net1(state, action)
            current_q2 = self.soft_q_net2(state, action)

            new_next_actions, new_log_pi= self.policy_net(next_state)

            target_q1 = self.target_soft_q_net1(next_state, new_next_actions)
            target_q2 = self.target_soft_q_net2(next_state, new_next_actions)
            target_q_values = torch.min(target_q1, target_q2) - alpha * new_log_pi

            q_target = reward + done * GAMMA * target_q_values
            q_loss = self.soft_q_criterion(current_q1, q_target.detach()) + self.soft_q_criterion(current_q2, q_target.detach())
            
            self.q_optimiser.zero_grad()
            q_loss.backward()
            self.q_optimiser.step()
           
    def update_alpha(self, state):
        new_actions, log_pi = self.policy_net(state)
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
                
    
     
     