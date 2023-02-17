import torch
import torch.optim as optim

from Components.Networks import SingleActor, SingleVNet
from Components.ReplayBuffers import OnPolicyBuffer

lr          = 3e-4
gamma=0.99


class A2C:
    def __init__(self, state_dim, n_acts) -> None:
        self.actor = SingleActor(state_dim, n_acts)
        self.critic = SingleVNet(state_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.replay_buffer = OnPolicyBuffer(state_dim, 10000)
        
    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor.pi(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action.numpy()
        
    def compute_rewards_to_go(self, rewards, done_masks):
        R = 0
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * done_masks[step]
            returns.insert(0, R)
            
        return returns
        
    def train(self, next_state=None):
        states, actions, next_states, rewards, done_masks = self.replay_buffer.make_data_batch()
        
        probs = self.actor.pi(states, softmax_dim=1)
        probs = probs.gather(1, actions.long())
        log_probs = torch.log(probs)[:, 0]

        returns = self.compute_rewards_to_go(rewards, done_masks)
        returns   = torch.cat(returns).detach()
        values    = self.critic.v(states)
        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    