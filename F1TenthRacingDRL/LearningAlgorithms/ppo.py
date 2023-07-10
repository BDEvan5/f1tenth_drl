import torch
import torch.optim as optim
import torch.nn.functional as F

from FTenthRacingDRL.Utils.Networks import SingleActor, SingleVNet
from FTenthRacingDRL.Utils.ReplayBuffers import OnPolicyBuffer



#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 100


class PPO:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        self.network = None
        self.optimizer = None

        self.actor = SingleActor(self.state_dim, self.action_dim)
        self.critic = SingleVNet(self.state_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate)
        
        self.replay_buffer = OnPolicyBuffer(state_dim, 10000)
        
    def act(self, obs):
        prob = self.actor.pi(torch.from_numpy(obs).float())
        m = torch.distributions.Categorical(prob)
        a = m.sample().item()

        return a
    
    def generalised_advantage_estimation(self, delta):
        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.FloatTensor(advantage_lst)
            
        return advantage
            
    def train(self, next_state=None):
        if self.replay_buffer.ptr < T_horizon:
            return

        states, actions, next_states, rewards, done_masks = self.replay_buffer.make_data_batch()

        for i in range(K_epoch):
            td_target = rewards + gamma * self.critic.v(next_states) * done_masks
            delta = td_target - self.critic.v(states)
            delta = delta.detach().numpy()

            advantage = self.generalised_advantage_estimation(delta)

            probs = self.actor.pi(states, softmax_dim=1)
            probs_for_actions = probs.gather(1,actions)
            # cloning and calling detatch() is how the surrogate objective calculated. Calling detach() on a tensor removes it from the gradient calculation
            prob_a = probs_for_actions.clone().detach()
            ratio = torch.exp(torch.log(probs_for_actions) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            # implementing the clipped surrogate objective
            surrogate_1 = ratio * advantage
            surrogate_2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surrogate_1, surrogate_2) + F.smooth_l1_loss(self.critic.v(states) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

