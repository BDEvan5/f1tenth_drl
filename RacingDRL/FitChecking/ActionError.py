import matplotlib.pyplot as plt
import numpy as np
import torch, os
from sklearn.metrics import mean_squared_error
from RacingDRL.FitChecking.train_networks import StdNetworkTwo

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

 
def load_data(folder, name):
    
    states = np.load(folder + f"DataSets/PurePursuit_{name}_states.npy")
    actions = np.load(folder + f"DataSets/PurePursuit_actions.npy")
    
    test_size = int(0.1*states.shape[0])
    test_inds = np.random.choice(states.shape[0], size=test_size, replace=False)
    
    test_x = torch.FloatTensor(states[test_inds])
    test_y = torch.FloatTensor(actions[test_inds])
    
    print(f"Test: {test_x.shape}")
    
    return test_x, test_y
    

def load_models(folder, ids, seed):
    models = []
    for id in ids:
        m = torch.load(folder + f"Models/endToEnd_{id}_{seed}.pt")
        models.append(m)
        
    return models

def run_all_seed():
    for i in range(5):
        run_action_analysis(i)

def run_action_analysis(seed):
    set_n = 3
    name = f"EndToEnd_stacking_{set_n}"
    folder = f"NetworkFitting/{name}/"
    ids = ["Single", "Double", "Triple", "Speed"]
    # ids = ["5", "10", "12", "15", "20", "30", "60"]
    
    np.random.seed(seed)
    test_sets = []
    for id in ids:
        x, y = load_data(folder, f"endToEnd_{id}")
        test_sets.append(x)
    true_actions = y.numpy()

    models = load_models(folder, ids, seed)
    
    result_data = []
    predicted_action_list = []
    plt.figure(figsize=(15, 6))
    for i in range(len(ids)):
        predicted_actions = models[i](test_sets[i]).detach().numpy()
        predicted_action_list.append(predicted_actions)
    
        # abs_losses = np.abs(true_actions - predicted_actions)
        abs_losses = (true_actions - predicted_actions)**2
        result_data.append(abs_losses)
        # mse_loss = mean_squared_error(true_actions, predicted_actions)
        # result_data.append(mse_loss)

    plt.clf()
    result_data = np.array(result_data)
    result_data = np.sqrt(result_data)
    print(result_data.shape)
    
    plt.boxplot(result_data[:, :, 0].T)
    plt.xticks(np.arange(len(ids)) + 1, ids)
    plt.grid(True)
    plt.title("Steering Error")
    plt.savefig(folder + f"Figures/BoxSteeringErrors_{seed}.svg")
    
    plt.clf()
    plt.boxplot(result_data[:, :, 1].T)
    plt.xticks(np.arange(len(ids)) + 1, ids)
    plt.grid(True)
    plt.title("Speed Error")
    plt.savefig(folder + f"Figures/BoxSpeedErrors_{seed}.svg")
    
    plt.clf()
    result_data = np.power(result_data, 2)
    total_loss = np.mean(result_data, axis=-1)
    total_loss = np.sqrt(total_loss)
    print(total_loss.shape)
    plt.boxplot(total_loss.T)
    plt.xticks(np.arange(len(ids)) + 1, ids)
    plt.grid(True)
    plt.title("Total Error")
    plt.savefig(folder + f"Figures/BoxTotalErrors_{seed}.svg")
    
    
    

# run_action_analysis()
run_all_seed()
