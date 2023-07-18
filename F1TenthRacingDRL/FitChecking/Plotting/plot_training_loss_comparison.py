import matplotlib.pyplot as plt
import numpy as np
from F1TenthRacingDRL.DataTools.plotting_utils import *
from matplotlib.ticker import MultipleLocator, MaxNLocator

  
def plot_training_testing_loss_comparison():
    experiment_name = "comparison"
    set_n = 1
    save_folder = f"TuningData/{experiment_name}_{set_n}/"

    name_keys = ["fullPlanning", "trajectoryTrack", "endToEnd"]
    label_names = ["Full planning", "Trajectory tracking", "End-to-end"]
    
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.2))
    
    mode = "train"
    xs = np.arange(0, 40) * 50
    
    for i, name in enumerate(name_keys):
        training_loss = np.load(save_folder + f"LossResults/{experiment_name}_{name}_train_losses.npy")[0:10]
        testing_loss = np.load(save_folder + f"LossResults/{experiment_name}_{name}_test_losses.npy")[0:10]

        axes[0].plot(xs, np.mean(training_loss, axis=0), label=label_names[i], color=color_pallet[i], alpha=0.9)
        axes[1].plot(xs, np.mean(testing_loss, axis=0), color=color_pallet[i], alpha=0.9)
            
    for i, name in enumerate(name_keys):
        training_loss = np.load(save_folder + f"LossResults/{experiment_name}_{name}_train_losses.npy")
        testing_loss = np.load(save_folder + f"LossResults/{experiment_name}_{name}_test_losses.npy")
        
        axes[0].fill_between(xs, np.min(training_loss, axis=0),  np.max(training_loss, axis=0), alpha=0.2, color=color_pallet[i])
        axes[1].fill_between(xs, np.min(testing_loss, axis=0),  np.max(testing_loss, axis=0), alpha=0.2, color=color_pallet[i])
        
    axes[0].set_xlabel("Training Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[1].set_xlabel("Training Epoch")
    axes[1].set_ylabel("Testing Loss")
    
    axes[0].set_ylim(0.03, 0.18)
    axes[1].set_ylim(0.03, 0.18)
    # axes[1].set_ylim(0.02, 0.11)
    
    # axes[0].xaxis.set_major_locator(MultipleLocator(500))
    axes[0].yaxis.set_major_locator(MultipleLocator(0.03))
    axes[1].yaxis.set_major_locator(MultipleLocator(0.03))
    
    handles, labels = axes[0].get_legend_handles_labels()
    print(labels)
    print(handles)
    fig.legend(labels[0:3], loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=3)
    # axes[0].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3)

    
    axes[0].grid(True)
    axes[1].grid(True)
    
    plt.tight_layout()
    
    plt.savefig(save_folder + f"LossResults/{experiment_name}_LossResults.svg", pad_inches=0, bbox_inches='tight')
    plt.savefig(save_folder + f"LossResults/{experiment_name}_LossResults.pdf", pad_inches=0, bbox_inches='tight')

plot_training_testing_loss_comparison()