import numpy as np
from matplotlib import pyplot as plt

from F1TenthRacingDRL.DataTools.plotting_utils import *
from matplotlib.ticker import MultipleLocator, MaxNLocator

 

def plot_n_beams_avg_seperate_normalised():
    name = "EndToEnd_stacking"
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
            # train_loss_mean.append(float(l[1]))
            # train_loss_std.append(float(l[2]))
            steer_loss_mean.append(float(l[5]))
            steer_loss_std.append(float(l[6]))
            speed_loss_mean.append(float(l[7]))
            speed_loss_std.append(float(l[8]))
            
    steer_loss_mean = np.array(steer_loss_mean)
    mean_steer_loss = np.mean(steer_loss_mean)
    steer_loss_mean = steer_loss_mean / mean_steer_loss
    steer_loss_std = np.array(steer_loss_std)/ mean_steer_loss

    speed_loss_mean = np.array(speed_loss_mean)
    mean_speed_loss = np.mean(speed_loss_mean)
    speed_loss_mean = speed_loss_mean / mean_speed_loss
    speed_loss_std = np.array(speed_loss_std)/ mean_speed_loss
    
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
    xs = np.arange(0, 5)
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]
    br2 = [x + barWidth for x in br1]
    
    plt.bar(br1, steer_loss_mean, color=pp_light[2], label="Steering", width=barWidth, alpha=0.7)
    plot_error_bars_single_colour(br1, steer_neg, steer_pos, pp_dark[2], w)
    plt.bar(br2, speed_loss_mean, color=pp_light[5], label="Speed", width=barWidth, alpha=0.7)
    plot_error_bars_single_colour(br2, speed_neg, speed_pos, pp_dark[5], w)
        
    # plt.xlabel("Method")
    name_values = [n.split("_")[1] for n in names]
    name_values[-2] = "Single \n+Speed"
    name_values[-1] = "Double \n+Speed"
    plt.xticks([r  for r in range(len(name_values))], name_values)
    plt.grid(True)
        
    plt.ylabel("Normalised Loss")
    plt.ylim(0.4, 1.55)
    plt.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.95))
    
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.3))
    
    plt.tight_layout()
    
    std_img_saving(path+ f"TrainTestLosses_seperate_{name}_normalised1")
    # plt.savefig(path + f"TrainTestLosses_seperate_{name}.svg")
    # plt.savefig(path + f"TrainTestLosses_seperate_{name}.pdf")
    


plot_n_beams_avg_seperate_normalised()
