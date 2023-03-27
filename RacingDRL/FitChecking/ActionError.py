import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from RacingDRL.FitChecking.CheckNetworkFit import StdNetworkTwo

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

folder  = "Data/PurePursuitDataGen_3/"

 
def load_data(name):
    
    states = np.load(folder + f"DataSets/PurePursuit_{name}_states.npy")
    actions = np.load(folder + f"DataSets/PurePursuit_actions.npy")
    
    np.random.seed(0)
    test_size = 200
    test_inds = np.random.choice(states.shape[0], size=test_size, replace=False)
    
    test_x = torch.FloatTensor(states[test_inds])
    test_y = torch.FloatTensor(actions[test_inds])
    
    print(f"Test: {test_x.shape}")
    
    return test_x, test_y
    

def load_models(ids):
    models = []
    for id in ids:
        m = torch.load(folder + f"Models/endToEnd_{id}.pt")
        models.append(m)
        
    return models

def run_action_analysis():
    ids = ["5", "10", "12", "15", "20", "30", "60"]
    test_sets = []
    for id in ids:
        x, y = load_data(f"endToEnd_{id}")
        test_sets.append(x)
    true_actions = y

    models = load_models(ids)
    
    rmse = []
    plt.figure(figsize=(10, 10))
    for i in range(len(ids)):
        predicted_actions = models[i](test_sets[i]).detach().numpy()
    
        mse_loss = mean_squared_error(true_actions, predicted_actions)
        rmse_loss = mse_loss ** 0.5
        print(f"{ids[i]}: {rmse_loss}")
        rmse.append(rmse_loss)
        
    plt.plot(ids, rmse)
        
    plt.xlabel("Number of beams")
    plt.ylabel("RMSE")
        
    plt.show()

run_action_analysis()

