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
    
def plot_n_beams_avg_seperate():
    name = "EndToEnd_stacking"
    steer_loss_mean, steer_loss_std = [], []
    speed_loss_mean, speed_loss_std = [], []
    train_loss_mean, train_loss_std = [], []
    names = []
    path = f"NetworkFitting/{name}_3/LossResultsSeperate/"
    with open(path + f"{name}_LossResultsSeperate.txt") as f:
        txt = f.readlines()
        for i, line in enumerate(txt):
            if i == 0: continue
            l = line.split(",")
            names.append(l[0])
            train_loss_mean.append(float(l[1]))
            train_loss_std.append(float(l[2]))
            steer_loss_mean.append(float(l[3]))
            steer_loss_std.append(float(l[4]))
            speed_loss_mean.append(float(l[5]))
            speed_loss_std.append(float(l[6]))
            
    train_loss_mean = np.array(train_loss_mean)
    train_loss_std = np.array(train_loss_std)
    steer_loss_mean = np.array(steer_loss_mean)
    steer_loss_std = np.array(steer_loss_std)
    speed_loss_mean = np.array(speed_loss_mean)
    speed_loss_std = np.array(speed_loss_std)
            
    print(names)
    
    print(steer_loss_mean)
    print(speed_loss_mean)
    
    # plt.figure(1, figsize=(4.2, 1.9))
    
    barWidth = 0.4
    w = 0.05
    xs = np.arange(0, 4)
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]
    br2 = [x + barWidth for x in br1]
    
    # train_pos = train_loss_mean + train_loss_std
    # train_neg = train_loss_mean - train_loss_std
    # test_pos = test_loss_mean + test_loss_std
    # test_neg = test_loss_mean - test_loss_std
    
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 1.9))
    axes[0].set_title("Steering")
    axes[1].set_title("Speed")
    # axes[0].set_ylim(0.06, 0.12)
    
    axes[0]
    axes[0].bar(br1, steer_loss_mean, color=pp_light[0], label="Train loss", width=barWidth)
    # plot_error_bars(br1, train_neg, train_pos, pp_dark[0], w)
    axes[1].bar(br1, speed_loss_mean, color=pp_light[1], label="Test loss", width=barWidth)
    # plot_error_bars(br2, test_neg, test_pos, pp_dark[1], w)
        
    # plt.xlabel("Method")
    name_values = [n.split("_")[1] for n in names]
    for i in range(2):
        plt.sca(axes[i])
        plt.xticks([r  for r in range(len(train_loss_mean))], name_values)
        plt.grid(True)
        
    # plt.ylabel("Loss (RMSE)")
    # plt.ylim(0.06, 0.12)
    # plt.legend(ncol=2)
    
    # plt.gca().yaxis.set_major_locator(MultipleLocator(0.01))
    
    plt.tight_layout()
    
    plt.savefig(path + f"TrainTestLosses_seperate_{name}.svg")
    plt.savefig(path + f"TrainTestLosses_seperate_{name}.pdf")
    

def plot_n_beams_avg_seperate_normalised():
    name = "EndToEnd_stacking"
    steer_loss_mean, steer_loss_std = [], []
    speed_loss_mean, speed_loss_std = [], []
    train_loss_mean, train_loss_std = [], []
    names = []
    path = f"NetworkFitting/{name}_3/LossResultsSeperate/"
    with open(path + f"{name}_LossResultsSeperate.txt") as f:
        txt = f.readlines()
        for i, line in enumerate(txt):
            if i == 0: continue
            l = line.split(",")
            names.append(l[0])
            train_loss_mean.append(float(l[1]))
            train_loss_std.append(float(l[2]))
            steer_loss_mean.append(float(l[3]))
            steer_loss_std.append(float(l[4]))
            speed_loss_mean.append(float(l[5]))
            speed_loss_std.append(float(l[6]))
            
    steer_loss_mean = np.array(steer_loss_mean)
    steer_loss_mean = steer_loss_mean / np.mean(steer_loss_mean)
    steer_loss_std = np.array(steer_loss_std)
    steer_loss_std = steer_loss_std / np.mean(steer_loss_mean)
    speed_loss_mean = np.array(speed_loss_mean)
    speed_loss_mean = speed_loss_mean / np.mean(speed_loss_mean)
    speed_loss_std = np.array(speed_loss_std)
    steer_loss_std = steer_loss_std / np.mean(speed_loss_mean)
    
    steer_neg = steer_loss_mean - steer_loss_std
    steer_pos = steer_loss_mean + steer_loss_std
    speed_neg = speed_loss_mean - speed_loss_std
    speed_pos = speed_loss_mean + speed_loss_std
            
    print(names)
    
    print(steer_loss_mean)
    print(speed_loss_mean)
    
    plt.figure(1, figsize=(4.2, 1.9))
    
    barWidth = 0.4
    w = 0.05
    xs = np.arange(0, 4)
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]
    br2 = [x + barWidth for x in br1]
    
    plt.bar(br1, steer_loss_mean, color=pp_light[2], label="Steering", width=barWidth)
    plot_error_bars(br1, steer_neg, steer_pos, pp_dark[2], w)
    plt.bar(br2, speed_loss_mean, color=pp_light[5], label="Speed", width=barWidth)
    plot_error_bars(br2, speed_neg, speed_pos, pp_dark[5], w)
        
    # plt.xlabel("Method")
    name_values = [n.split("_")[1] for n in names]
    plt.xticks([r  for r in range(len(train_loss_mean))], name_values)
    plt.grid(True)
        
    plt.ylabel("Normalised Loss")
    plt.ylim(0.7, 1.15)
    plt.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.95))
    
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    
    plt.tight_layout()
    
    std_img_saving(path+ f"TrainTestLosses_seperate_{name}_normalised")
    # plt.savefig(path + f"TrainTestLosses_seperate_{name}.svg")
    # plt.savefig(path + f"TrainTestLosses_seperate_{name}.pdf")
    

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
# plot_n_beams_avg()
# plot_n_beams_avg_seperate()
plot_n_beams_avg_seperate_normalised()
