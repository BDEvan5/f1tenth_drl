import matplotlib.pyplot as plt
import numpy as np
from RacingDRL.DataTools.plotting_utils import *

from matplotlib.ticker import MultipleLocator

def plot_simple_beam_example():
    experiment_name = "comparison"
    # name = "fullPlanning"
    name = "trajectoryTrack"
    name = "endToEnd"
    # experiment_name = "EndToEnd_nBeams"
    # name = "endToEnd_20"
    set_n = 3
    save_folder = f"NetworkFitting/{experiment_name}_{set_n}/"

    
    # plt.figure(figsize=(4, 2))
    plt.figure(figsize=(5, 2.8))
    training_loss = np.load(save_folder + f"LossResults/{experiment_name}_{name}_train_losses.npy")
    plt.plot(np.mean(training_loss, axis=0), color=pp[0], label="Training loss")
    # plt.fill_between(np.arange(len(training_loss[0])), np.min(training_loss, axis=0),  np.max(training_loss, axis=0), alpha=0.2, color=pp[1])
        
    
    test_loss = np.load(save_folder + f"LossResults/{experiment_name}_{name}_test_losses.npy")
    plt.plot(np.mean(test_loss, axis=0), color=pp[1], label="Test loss")
    # plt.fill_between(np.arange(len(test_loss[0])), np.min(test_loss, axis=0),  np.max(test_loss, axis=0), alpha=0.2, color=pp[0])   
            
     
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")
    
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(save_folder + f"LossResults/{name}_SimpleLossResults.svg", pad_inches=0, bbox_inches='tight')
    plt.savefig(save_folder + f"LossResults/{name}_SimpleLossResults.pdf", pad_inches=0, bbox_inches='tight')
    
    # plt.show()
    
plot_simple_beam_example()