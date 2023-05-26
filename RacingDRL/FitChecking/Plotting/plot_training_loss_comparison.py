import matplotlib.pyplot as plt
import numpy as np
from RacingDRL.DataTools.plotting_utils import *
from matplotlib.ticker import MultipleLocator, MaxNLocator


def plot_training_loss_comparison():
    experiment_name = "comparison"
    set_n = 3
    save_folder = f"NetworkFitting/{experiment_name}_{set_n}/"

    name_keys = ["fullPlanning", "trajectoryTrack", "endToEnd"]
    label_names = ["Full Planning", "Trajectory Tracking", "End-to-end"]
    # name_keys = ["fullPlanning", "trajectoryTrack", "endToEnd", "endToEnd_Single"]
    # label_names = ["fullPlanning", "trajectoryTrack", "endToEnd", "endToEnd_Single"]
    
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.2))
    
    mode = "train"
    xs = np.arange(0, 41) * 50
    
    for i, name in enumerate(name_keys):
        speed_training_loss = np.load(save_folder + f"LossResultsSeperate/{experiment_name}_{name}_test_losses_speed.npy")
        steer_training_loss = np.load(save_folder + f"LossResultsSeperate/{experiment_name}_{name}_test_losses_steering.npy")

        axes[0].plot(xs, np.mean(speed_training_loss, axis=0), label=label_names[i], color=color_pallet[i], alpha=0.9)
        axes[1].plot(xs, np.mean(steer_training_loss, axis=0), color=color_pallet[i], alpha=0.8)
            
    for i, name in enumerate(name_keys):
        speed_training_loss = np.load(save_folder + f"LossResultsSeperate/{experiment_name}_{name}_test_losses_speed.npy")
        steer_training_loss = np.load(save_folder + f"LossResultsSeperate/{experiment_name}_{name}_test_losses_steering.npy")
        
        axes[0].fill_between(xs, np.min(speed_training_loss, axis=0),  np.max(speed_training_loss, axis=0), alpha=0.2, color=color_pallet[i])
        axes[1].fill_between(xs, np.min(steer_training_loss, axis=0),  np.max(steer_training_loss, axis=0), alpha=0.2, color=color_pallet[i])
        
    axes[0].set_xlabel("Training Epoch")
    axes[0].set_ylabel("Speed Loss")
    axes[1].set_xlabel("Training Epoch")
    axes[1].set_ylabel("Steering Loss")
    
    axes[0].set_ylim(0, 0.08)
    axes[1].set_ylim(0.02, 0.11)
    
    # axes[0].xaxis.set_major_locator(MultipleLocator(500))
    axes[1].yaxis.set_major_locator(MultipleLocator(0.02))
    
    handles, labels = axes[0].get_legend_handles_labels()
    print(labels)
    print(handles)
    fig.legend(labels[0:3], loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=3)
    # axes[0].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3)

    
    axes[0].grid(True)
    axes[1].grid(True)
    
    plt.tight_layout()
    
    plt.savefig(save_folder + f"LossResultsSeperate/{experiment_name}_LossResultsSeperate.svg", pad_inches=0, bbox_inches='tight')
    plt.savefig(save_folder + f"LossResultsSeperate/{experiment_name}_LossResultsSeperate.pdf", pad_inches=0, bbox_inches='tight')
    
def plot_training_testing_loss_comparison():
    experiment_name = "comparison"
    set_n = 3
    save_folder = f"NetworkFitting/{experiment_name}_{set_n}/"

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
    
    axes[0].set_ylim(0.03, 0.11)
    axes[1].set_ylim(0.03, 0.11)
    # axes[1].set_ylim(0.02, 0.11)
    
    # axes[0].xaxis.set_major_locator(MultipleLocator(500))
    axes[1].yaxis.set_major_locator(MultipleLocator(0.02))
    
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
    
def plot_training__combined_loss_comparison():
    experiment_name = "comparison"
    set_n = 3
    save_folder = f"NetworkFitting/{experiment_name}_{set_n}/"

    name_keys = ["fullPlanning", "trajectoryTrack", "endToEnd"]
    label_names = ["Full planning", "Trajectory tracking", "End-to-end"]
    # name_keys = ["fullPlanning", "trajectoryTrack", "endToEnd", "endToEnd_Single"]
    # label_names = ["fullPlanning", "trajectoryTrack", "endToEnd", "endToEnd_Single"]
    
    plt.figure(figsize=(5, 2.2))
    
    for i, name in enumerate(name_keys):
        training_loss = np.load(save_folder + f"LossResultsSeperate/{experiment_name}_{name}_train_losses.npy")

        plt.plot(np.mean(training_loss, axis=0), color=pp[i+1], label=label_names[i])
            
    for i, name in enumerate(name_keys):
        training_loss = np.load(save_folder + f"LossResultsSeperate/{experiment_name}_{name}_train_losses.npy")
        plt.fill_between(np.arange(len(training_loss[0])), np.min(training_loss, axis=0),  np.max(training_loss, axis=0), alpha=0.2, color=pp[i+1])
        
    plt.xlabel("Training Epoch")
    plt.ylabel("Speed Loss")
    
    # axes[0].set_ylim(0, 0.3)
    
    # plt.gca().xaxis.set_ticks(np.arange(0, 300, 80))
    # plt.gca().yaxis.set_ticks(np.arange(0, 0.8, 0.05))
    
    # labels = plt.get_figlabels()
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    plt.legend()
    
    plt.grid(True)
    
    plt.tight_layout()
    
    plt.savefig(save_folder + f"LossResultsSeperate/{experiment_name}_LossResults.svg", pad_inches=0, bbox_inches='tight')
    plt.savefig(save_folder + f"LossResultsSeperate/{experiment_name}_LossResults.pdf", pad_inches=0, bbox_inches='tight')
    
    
# plot_training_loss_comparison()
plot_training_testing_loss_comparison()
# plot_training__combined_loss_comparison()