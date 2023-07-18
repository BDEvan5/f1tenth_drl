
import matplotlib.pyplot as plt
import numpy as np
from F1TenthRacingDRL.DataTools.plotting_utils import *
from matplotlib.ticker import MultipleLocator


set_n = 1

def TuningData_loss_graphs():
    name = "EndToEnd_nBeams"
    test_loss_mean, test_loss_std = [], []
    train_loss_mean, train_loss_std = [], []
    names = []
    path = f"TuningData/EndToEnd_nBeams_{set_n}/LossResults/"
    with open(path + f"{name}_LossResults.txt") as f:
        txt = f.readlines()
        for i, line in enumerate(txt):
            if i == 0: continue
            l = line.split(",")
            names.append(l[0])
            train_loss_mean.append(float(l[1]))
            train_loss_std.append(float(l[2]))
            test_loss_mean.append(float(l[3]))
            test_loss_std.append(float(l[4]))
            
    train_loss_mean = np.array(train_loss_mean)
    train_loss_std = np.array(train_loss_std)
    test_loss_mean = np.array(test_loss_mean)
    test_loss_std = np.array(test_loss_std)
            
    name_values = [int(n.split("_")[1]) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(5.8,2), sharey=True, sharex=False)

    axes[0].plot(name_values, train_loss_mean, '.-', color=pp[0], label="Training", markersize=10, linewidth=2)
    axes[0].plot(name_values, test_loss_mean, '.-', color=pp[1], label="Testing", markersize=10, linewidth=2)
    
    train_pos = train_loss_mean + train_loss_std
    train_neg = train_loss_mean - train_loss_std
    test_pos = test_loss_mean + test_loss_std
    test_neg = test_loss_mean - test_loss_std
    
    axes[0].fill_between(name_values, train_pos, train_neg, color=pp[0], alpha=0.25)
    axes[0].fill_between(name_values, test_pos, test_neg, color=pp[1], alpha=0.15)
    
    axes[0].set_xlabel("Number of beams")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].xaxis.set_major_locator(MultipleLocator(10))
    
    name = "trajectoryTrack_nWaypoints"
    test_loss_mean, test_loss_std = [], []
    train_loss_mean, train_loss_std = [], []
    names = []
    path = f"TuningData/{name}_{set_n}/LossResults/"
    with open(path + f"{name}_LossResults.txt") as f:
        txt = f.readlines()
        for i, line in enumerate(txt):
            if i == 0: continue
            l = line.split(",")
            names.append(l[0])
            train_loss_mean.append(float(l[1]))
            train_loss_std.append(float(l[2]))
            test_loss_mean.append(float(l[3]))
            test_loss_std.append(float(l[4]))
            
    train_loss_mean = np.array(train_loss_mean)
    train_loss_std = np.array(train_loss_std)
    test_loss_mean = np.array(test_loss_mean)
    test_loss_std = np.array(test_loss_std)
            
    name_values = [int(n.split("_")[1]) for n in names]
    
    axes[1].plot(name_values, train_loss_mean, '.-', color=pp[0], label="Train loss", markersize=10, linewidth=2)
    axes[1].plot(name_values, test_loss_mean, '.-', color=pp[1], label="Test loss", markersize=10, linewidth=2)
    
    train_pos = train_loss_mean + train_loss_std
    train_neg = train_loss_mean - train_loss_std
    test_pos = test_loss_mean + test_loss_std
    test_neg = test_loss_mean - test_loss_std
    
    axes[1].fill_between(name_values, train_pos, train_neg, color=pp[0], alpha=0.25)
    axes[1].fill_between(name_values, test_pos, test_neg, color=pp[1], alpha=0.15)
    
    axes[1].set_xlabel("Number of waypoints")
    # axes[1].set_ylabel("Loss (RMSE)")
    axes[1].yaxis.set_major_locator(MultipleLocator(0.03))
    axes[1].grid(True)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.92))
    
    name = f"TuningData/TuningDataGraph"
    std_img_saving(name)

TuningData_loss_graphs()