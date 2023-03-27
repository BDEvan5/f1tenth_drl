import matplotlib.pyplot as plt
import numpy as np
from RacingDRL.DataTools.plotting_utils import *

def plot_n_beams():
    test_losses = []
    train_losses = []
    names = []
    with open("Data/PurePursuitDataGen_3/LossImgs/LossResults.txt") as f:
        txt = f.readlines()
        for i, line in enumerate(txt):
            if i == 0: continue
            l = line.split(",")
            names.append(l[0])
            train_losses.append(float(l[1]))
            test_losses.append(float(l[2]))
            
    print(names)
    name_values = [int(n.split("_")[1]) for n in names]
    print(train_losses)
    print(test_losses)
    
    plt.figure(1, figsize=(4,2))
    
    plt.plot(name_values, train_losses, '.-', color=pp[0], label="Train loss", markersize=10, linewidth=2)
    plt.plot(name_values, test_losses, '.-', color=pp[1], label="Test loss", markersize=10, linewidth=2)
    
    plt.xlabel("Number of beams")
    plt.ylabel("Loss (RMSE)")
    plt.ylim(0.075, 0.14)
    plt.legend()
    
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig("Data/PurePursuitDataGen_3/LossImgs/TrainTestLosses.svg")
    plt.savefig("Data/PurePursuitDataGen_3/LossImgs/TrainTestLosses.pdf")
    
    
plot_n_beams()
