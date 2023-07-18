import torch
import torch.optim as optim
import random

from F1TenthRacingDRL.Utils.Networks import QNetworkDQN
from F1TenthRacingDRL.Utils.ReplayBuffers import OffPolicyBuffer

GAMMA = 0.94
LEARNING_RATE = 0.001

MEMORY_SIZE = 100000
BATCH_SIZE = 32

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQN:
    def __init__(self, obs_space, action_space):
        self.obs_space = obs_space
        self.action_space = action_space
        self.replay_buffer = OffPolicyBuffer(obs_space, 1)

        self.model = QNetworkDQN(obs_space, action_space)
        self.target = QNetworkDQN(obs_space, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.exploration_rate = EXPLORATION_MAX
        self.update_steps = 0

    def act(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_space-1)
        else: 
            state = torch.from_numpy(state).float()
            q_values = self.model.forward(state)
            action = q_values.argmax().item() 
            return action
        
    def train(self):
        if self.replay_buffer.size() < BATCH_SIZE: return
        state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
        
        next_values = self.target.forward(next_state)
        max_vals = torch.max(next_values, dim=1)[0].reshape((BATCH_SIZE, 1))
        q_target = reward + GAMMA * max_vals * done
        q_vals = self.model.forward(state)
        current_q_a = q_vals.gather(1, action.type(torch.int64))
        
        loss = torch.nn.functional.mse_loss(current_q_a, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % 100 == 1: 
            self.target.load_state_dict(self.model.state_dict())
        if self.update_steps % 12 == 1:
            self.exploration_rate *= EXPLORATION_DECAY 
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
            
        if self.update_steps % 1000 == 1:
            print("Exploration rate: ", self.exploration_rate)

