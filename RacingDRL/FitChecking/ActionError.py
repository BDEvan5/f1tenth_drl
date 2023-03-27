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
    true_actions = y.numpy()

    models = load_models(ids)
    
    data_speed = []
    data_steer = []
    predicted_action_list = []
    plt.figure(figsize=(15, 6))
    for i in range(len(ids)):
        predicted_actions = models[i](test_sets[i]).detach().numpy()
        predicted_action_list.append(predicted_actions)
    
        abs_losses = np.abs(true_actions - predicted_actions)
        data_speed.append(abs_losses)

        plt.plot(predicted_actions[:, 1], label=ids[i])
    plt.plot(true_actions[:, 1], label=True, linewidth=2)
    
    plt.xlim(0, 60)
    plt.grid()
    plt.tight_layout()
    plt.legend(ncol=len(ids))
    plt.savefig(folder + "Figures/SpeedError.svg")
    
    plt.clf()
    for i in range(len(ids)):
        predicted_actions = models[i](test_sets[i]).detach().numpy()
        predicted_action_list.append(predicted_actions)
    
        abs_losses = np.abs(true_actions - predicted_actions)
        data_steer.append(abs_losses)

        plt.plot(predicted_actions[:, 0], label=ids[i])
    plt.plot(true_actions[:, 0], label="True", linewidth=2)
    
    plt.xlim(0, 60)
    plt.grid()
    plt.tight_layout()
    plt.legend(ncol=len(ids))
        
    plt.savefig(folder + "Figures/SteeringError.svg")

    plt.clf()
    data_steer = np.array(data_steer)
    print(data_steer.shape)
    data_speed = np.array(data_speed)
    print(data_speed.shape)
    # labels = ids.append("True")
    
    
    plt.boxplot(data_steer[:, :, 0].T)
    plt.xticks(np.arange(7) + 1, ids)
    plt.xlabel("Number of beams")
    plt.grid(True)
    plt.ylabel("Steering Error")
    plt.savefig(folder + "Figures/BoxSteeringErrors.svg")
    
    plt.clf()
    plt.boxplot(data_steer[:, :, 1].T)
    plt.xticks(np.arange(7) + 1, ids)
    plt.xlabel("Number of beams")
    plt.grid(True)
    plt.ylabel("Speed Error")
    plt.savefig(folder + "Figures/BoxSpeedErrors.svg")
    
    
    

run_action_analysis()

