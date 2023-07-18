import matplotlib.pyplot as plt
import numpy as np
from F1TenthRacingDRL.DataTools.plotting_utils import *
from matplotlib.ticker import MultipleLocator
import pandas as pd


def plot_n_beams_avg():
    name = "TuningNN"
    test_loss_mean, test_loss_std = [], []
    train_loss_mean, train_loss_std = [], []
    names = []
    path = "TuningData/TuningNN_1/LossResultsE/"
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
    name_values = [int(n) for n in names]
    # name_values = [int(n.split("_")[1]) for n in names]
    # print(train_losses)
    # print(test_losses)

    df = pd.DataFrame([name_values, train_loss_mean, train_loss_std, test_loss_mean, test_loss_std]).T
    df.columns = ["Neurons", "TrainLoss", "TrainLossStd", "TestLoss", "TestLossStd"]
    df = df.sort_values(by="Neurons")
    
    plt.figure(1, figsize=(4,2))
    
    plt.plot(df["Neurons"], df["TrainLoss"], '.-', color=pp[0], label="Training", markersize=10, linewidth=2)
    plt.plot(df["Neurons"], df["TestLoss"], '.-', color=pp[1], label="Testing", markersize=10, linewidth=2)    
    
    df["train_low"] = df["TrainLoss"] - df["TrainLossStd"]
    df["train_high"] = df["TrainLoss"] + df["TrainLossStd"]
    df["test_low"] = df["TestLoss"] - df["TestLossStd"]
    df["test_high"] = df["TestLoss"] + df["TestLossStd"]
    
    plt.fill_between(df["Neurons"], df["train_low"], df["train_high"], color=pp[0], alpha=0.25)
    plt.fill_between(df["Neurons"], df["test_low"], df["test_high"], color=pp[1], alpha=0.15)
    
    plt.xlabel("Number of Neurons")
    plt.ylabel("Loss (RMSE)")
    plt.ylim(0.06, 0.17)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.02))
    # plt.ylim(0.075, 0.14)
    plt.legend(ncol=2)
    
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(path + f"TrainTestLosses_{name}.svg")
    plt.savefig(path + f"TrainTestLosses_{name}.pdf")
    
    

def plot_n_beams_avg_small():
    name = "EndToEnd_nBeams"
    test_loss_mean, test_loss_std = [], []
    train_loss_mean, train_loss_std = [], []
    names = []
    path = "TuningData/EndToEnd_nBeams_3/LossResults/"
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
    # print(train_losses)
    # print(test_losses)
    
    # plt.figure(1, figsize=(4,2))
    plt.figure(1, figsize=(3,2))
    
    plt.plot(name_values, train_loss_mean, '.-', color=pp[0], label="Train loss", markersize=10, linewidth=2)
    plt.plot(name_values, test_loss_mean, '.-', color=pp[1], label="Test loss", markersize=10, linewidth=2)
    
    train_pos = train_loss_mean + train_loss_std
    train_neg = train_loss_mean - train_loss_std
    test_pos = test_loss_mean + test_loss_std
    test_neg = test_loss_mean - test_loss_std
    
    plt.fill_between(name_values, train_pos, train_neg, color=pp[0], alpha=0.3)
    plt.fill_between(name_values, test_pos, test_neg, color=pp[1], alpha=0.3)
    
    plt.xlabel("Number of beams")
    plt.ylabel("Loss (RMSE)")
    plt.ylim(0.05, 0.14)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.02))
    # plt.ylim(0.075, 0.14)
    plt.legend()
    
    plt.grid(True)
    plt.tight_layout()
    
    name = path + f"TrainTestLosses_{name}_small"
    std_img_saving(name)
    
    
plot_n_beams_avg()
# plot_n_beams_avg_small()