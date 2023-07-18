from F1TenthRacingDRL.Utils.Networks import DoublePolicyNet, DoubleQNet
from F1TenthRacingDRL.Utils.ReplayBuffers import OffPolicyBuffer
from F1TenthRacingDRL.Utils.utils import soft_update

import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
BATCH_SIZE   = 32
tau          = 0.005 # for target network soft update


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
  
class TrainDDPG:
    def __init__(self, state_dim, action_dim):
        self.replay_buffer = OffPolicyBuffer(state_dim, action_dim)

        self.critic = DoubleQNet(state_dim, action_dim)
        self.critic_target = DoubleQNet(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor = DoublePolicyNet(state_dim, action_dim)
        self.actor_target = DoublePolicyNet(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.mu_optimizer = optim.Adam(self.actor.parameters(), lr=lr_mu)
        self.q_optimizer  = optim.Adam(self.critic.parameters(), lr=lr_q)
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    def act(self, state):
        action = self.actor(torch.from_numpy(state).float()) 
        action = action.detach().numpy() + self.ou_noise()
        
        return action
      
    def train(self):
        if self.replay_buffer.size() < BATCH_SIZE: return
        
        states, actions, next_states, rewards, done_masks  = self.replay_buffer.sample(BATCH_SIZE)
        
        target = rewards + gamma * self.critic_target(next_states, self.actor_target(next_states)) * done_masks
        q_loss = F.smooth_l1_loss(self.critic(states,actions), target.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        mu_loss = -self.critic(states, self.actor(states)).mean() 
        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()
    
        soft_update(self.critic, self.critic_target, tau)
        soft_update(self.actor, self.actor_target, tau)
     
    def save(self, filename, directory):
        torch.save(self.actor, '%s/%s_actor.pth' % (directory, filename))
     
     
class TestDDPG:
    def __init__(self, filename, directory):
        self.actor = torch.load('%s/%s_actor.pth' % (directory, filename))

    def act(self, state):
        state = torch.FloatTensor(state)
        action = self.actor(state).data.numpy().flatten()
        
        return action
      
        