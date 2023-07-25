import matplotlib.pyplot as plt
import numpy as np
from F1TenthRacingDRL.DataTools.plotting_utils import *
from matplotlib.ticker import MultipleLocator


def plot_n_waypoints_avg():
    set_n = 3
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
            
    print(names)
    name_values = [int(n.split("_")[1]) for n in names]
    
    plt.figure(1, figsize=(4,2))
    
    plt.plot(name_values, train_loss_mean, '.-', color=pp[0], label="Train loss", markersize=10, linewidth=2)
    plt.plot(name_values, test_loss_mean, '.-', color=pp[1], label="Test loss", markersize=10, linewidth=2)
    
    train_pos = train_loss_mean + train_loss_std
    train_neg = train_loss_mean - train_loss_std
    test_pos = test_loss_mean + test_loss_std
    test_neg = test_loss_mean - test_loss_std
    
    plt.fill_between(name_values, train_pos, train_neg, color=pp[0], alpha=0.3)
    plt.fill_between(name_values, test_pos, test_neg, color=pp[1], alpha=0.3)
    
    plt.xlabel("Number of waypoints")
    plt.ylabel("Loss (RMSE)")
    # plt.ylim(0.075, 0.14)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.02))
    plt.legend()
    
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(path + f"TrainTestLosses_{name}.svg")
    plt.savefig(path + f"TrainTestLosses_{name}.pdf")
    
    
plot_n_waypoints_avg()