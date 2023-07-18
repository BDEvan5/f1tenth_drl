import numpy as np
from matplotlib import pyplot as plt

from F1TenthRacingDRL.DataTools.plotting_utils import *
from matplotlib.ticker import MultipleLocator, MaxNLocator



def plot_separate_normalised_loss_new():
    name = "fullPlanning_ablation"
    steer_loss_mean, steer_loss_std = [], []
    speed_loss_mean, speed_loss_std = [], []
    train_loss_mean, train_loss_std = [], []
    names = []
    path = f"TuningData/{name}_3/LossResultsSeperate/"
    with open(path + f"{name}_LossResultsSeperate.txt") as f:
        txt = f.readlines()
        for i, line in enumerate(txt):
            if i == 0: continue
            l = line.split(",")
            names.append(l[0])
            steer_loss_mean.append(float(l[5]))
            steer_loss_std.append(float(l[6]))
            speed_loss_mean.append(float(l[7]))
            speed_loss_std.append(float(l[8]))
            
    steer_loss_mean = np.array(steer_loss_mean)
    mean_steer_loss = np.mean(steer_loss_mean)
    steer_loss_mean = steer_loss_mean / mean_steer_loss
    steer_loss_std = np.array(steer_loss_std)
    steer_loss_std = steer_loss_std / mean_steer_loss
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

    print(steer_neg)
    print(steer_loss_std)
    print(steer_pos)

    plt.figure(1, figsize=(4.2, 2))
    
    barWidth = 0.4
    w = 0.05
    xs = np.arange(0, len(steer_loss_mean))
    br1 = xs - barWidth / 2
    br2 = [x + barWidth for x in br1]
    
    plt.bar(br1, steer_loss_mean, color=pp_light[2], label="Steering", width=barWidth, alpha=0.7)
    plot_error_bars_single_colour(br1, steer_neg, steer_pos, pp_dark[2], w)
    plt.bar(br2, speed_loss_mean, color=pp_light[5], label="Speed", width=barWidth, alpha=0.7)
    plot_error_bars_single_colour(br2, speed_neg, speed_pos, pp_dark[5], w)
        
    new_name_values = ["Motion +\n LiDAR +\n Waypoints", "LiDAR +\n Waypoints", "Motion +\n Waypoints", "Motion +\n LiDAR", "Motion"]
    plt.xticks([r  for r in range(len(steer_loss_mean))], new_name_values, size=7)
    plt.grid(True)
        
    plt.ylabel("Normalised Loss")
    plt.ylim(0.5, 1.8)
    plt.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.95))
    
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.3))
    
    plt.tight_layout()
    
    std_img_saving(path+ f"TrainTestLosses_seperate_{name}_normalised_new")
    # plt.savefig(path + f"TrainTestLosses_seperate_{name}.svg")
    # plt.savefig(path + f"TrainTestLosses_seperate_{name}.pdf")
    

plot_separate_normalised_loss_new()
