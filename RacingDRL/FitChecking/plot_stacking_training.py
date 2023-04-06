import numpy as np
from matplotlib import pyplot as plt

from RacingDRL.DataTools.plotting_utils import *
from matplotlib.ticker import MultipleLocator, MaxNLocator

def plot_n_beams_avg():
    name = "EndToEnd_stacking"
    test_loss_mean, test_loss_std = [], []
    train_loss_mean, train_loss_std = [], []
    names = []
    path = f"NetworkFitting/{name}_3/LossResults/"
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
    
    plt.figure(1, figsize=(4.2, 1.9))
    
    barWidth = 0.4
    w = 0.05
    xs = np.arange(0, 4)
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]
    
    train_pos = train_loss_mean + train_loss_std
    train_neg = train_loss_mean - train_loss_std
    test_pos = test_loss_mean + test_loss_std
    test_neg = test_loss_mean - test_loss_std
    
    plt.bar(br1, train_loss_mean, color=pp_light[0], label="Train loss", width=barWidth)
    plot_error_bars(br1, train_neg, train_pos, pp_dark[0], w)
    plt.bar(br2, test_loss_mean, color=pp_light[1], label="Test loss", width=barWidth)
    plot_error_bars(br2, test_neg, test_pos, pp_dark[1], w)
        
    # plt.xlabel("Method")
    name_values = [n.split("_")[1] for n in names]
    plt.xticks([r  for r in range(len(train_loss_mean))], name_values)
    plt.ylabel("Loss (RMSE)")
    plt.ylim(0.06, 0.12)
    plt.legend(ncol=2)
    
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.01))
    
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(path + f"TrainTestLosses_{name}.svg")
    plt.savefig(path + f"TrainTestLosses_{name}.pdf")
    

def plot_stacking_training():
    name = "EndToEnd_stacking"
    set_n = 3
    path = f"NetworkFitting/{name}_{set_n}/LossResults/"
    name_keys = ["endToEnd_Single", "endToEnd_Double", "endToEnd_Triple", "endToEnd_Speed"]
    
    fig, axs = plt.subplots(1, 2, figsize=(5, 3))
    
    for k, key in enumerate(name_keys):
        test_losses = np.load(path + name + "_" + key + "_test_losses.npy")
        train_losses = np.load(path + name + "_" + key + "_train_losses.npy")
        
        # for i in range(len(test_losses)):
        for i in range(1):
            axs[0].plot(train_losses[i], color=pp[k], alpha=0.5)
            axs[1].plot(test_losses[i], color=pp[k], alpha=0.5)
        
    for a in range(2):
        axs[a].grid(True)
        axs[a].set_xlabel("Epoch")
    
    axs[0].set_title("Train loss")
    axs[1].set_title("Test loss")
    
    plt.tight_layout()
    
    plt.savefig(path + f"LossPlot_{name}.svg")
    plt.savefig(path + f"LossPlot_{name}.pdf")
        
# plot_stacking_training()
plot_n_beams_avg()
