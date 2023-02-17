from RacingDRL.LearningAlgorithms.td3 import TD3
from RacingDRL.LearningAlgorithms.sac import SAC
from RacingDRL.LearningAlgorithms.ddpg import DDPG
from RacingDRL.LearningAlgorithms.ppo import PPO
from RacingDRL.LearningAlgorithms.dqn import DQN
from RacingDRL.LearningAlgorithms.a2c import A2C


def create_agent(run_dict, state_dim):
    action_dim = 2
    
    if run_dict.algorithm == "TD3":
        agent = TD3(state_dim, action_dim)
    elif run_dict.algorithm == "SAC":
        agent = SAC(state_dim, action_dim)
    elif run_dict.algorithm == "DDPG":
        agent = DDPG(state_dim, action_dim)
    # elif run_dict.algorithm == "PPO":
    #     agent = PPO(state_dim, action_dim)
    # elif run_dict.algorithm == "DQN":
    #     agent = DQN(state_dim, action_dim)
    # elif run_dict.algorithm == "A2C":
    #     agent = A2C(state_dim, action_dim)
    else: raise ValueError(f"Algorithm {run_dict.algorithm} not recognised")
    
    return agent
    