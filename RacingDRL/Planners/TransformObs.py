

def end_to_end_obs(obs):
    state = obs['scans'][0]
    
    return state

def path_follower_obs(obs):
    raise NotImplementedError

def hybrid_obs(obs):
    raise NotImplementedError