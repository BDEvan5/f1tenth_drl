from matplotlib import pyplot as plt
import numpy as np
from F1TenthRacingDRL.DataTools.plotting_utils import *
import pandas as pd 

def make_deviation_bar_plot():
    map_name = "mco"
    set_number = 1
    base_path = f"Data/FinalExperiment_{set_number}/"
    vehicle_keys = ["Game", "TrajectoryFollower", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]
    iteration = 0
    
    folder_list = [base_path + f"AgentOff_SAC_{vehicle_keys[i]}_{map_name}_TAL_8_{set_number}_{iteration}/" for i in range(3)]
    folder_list.append(base_path + f"PurePursuit_PP_pathFollower_{map_name}_TAL_8_{set_number}_0/")
    
    df = pd.read_csv(base_path + f"ExtraData.csv").fillna(0)
    df = df[df.TestMap == "GBR"]
    df = df[(df.Algorithm == "SAC") | (df.Algorithm == "PP")]
    df = df[(df.TrainID != "Progress")]
    df = df[(df.Repetition == 1) |  (df.Algorithm == "PP")]
    # df = df[df.Repetition == 0]
    print(df)

    xs = np.arange(4)
    fig, axs = plt.subplots(2, 2, figsize=(5, 3), sharey='row')
    for z, reward  in enumerate(df.TrainID.unique()):
        mini_df = df[(df.TrainID == reward) | (df.Algorithm == "PP")]
        print(mini_df)
        axs[0, z].bar(xs, mini_df.LateralD_M*100, capsize=100, color=color_pallet, alpha=0.5)
        plt.sca(axs[0, z])
        plot_error_bars(xs, mini_df.LateralD_Q1.to_numpy()*100, mini_df.LateralD_Q3.to_numpy()*100, color_pallet, w=0.5, tails=False)
        axs[0, z].set_ylim([0, 50])
        
        axs[1, z].bar(xs, mini_df.SpeedD_M,  capsize=10, color=color_pallet, alpha=0.5)
        plt.sca(axs[1, z])
        plot_error_bars(xs, mini_df.SpeedD_Q1.to_numpy(), mini_df.SpeedD_Q3.to_numpy(), color_pallet, w=0.5, tails=False)
        axs[1, z].set_ylim([0, 3])
        
        axs[z, 0].grid(True)
        axs[z, 1].grid(True)
        axs[z, 0].set_xticks([])
        axs[z, 1].set_xticks([])

    x_val = 2.6
    y_val = 40
    axs[0, 0].text(x_val, y_val, "CTH", fontdict={'fontsize': 10, 'fontweight':'bold'}, bbox=dict(facecolor='white', alpha=0.99, boxstyle='round,pad=0.15', edgecolor='gainsboro'))
    axs[0, 1].text(x_val, y_val, "TAL", fontdict={'fontsize': 10, 'fontweight':'bold'}, bbox=dict(facecolor='white', alpha=0.99, boxstyle='round,pad=0.15', edgecolor='gainsboro'))
    y_val = 2.4
    axs[1, 0].text(x_val, y_val, "CTH", fontdict={'fontsize': 10, 'fontweight':'bold'}, bbox=dict(facecolor='white', alpha=0.99, boxstyle='round,pad=0.15', edgecolor='gainsboro'))
    axs[1, 1].text(x_val, y_val, "TAL", fontdict={'fontsize': 10, 'fontweight':'bold'}, bbox=dict(facecolor='white', alpha=0.99, boxstyle='round,pad=0.15', edgecolor='gainsboro'))

    # axs[0, 0].set_title("CTH")
    # axs[0, 1].set_title("TAL")

    axs[0, 0].set_ylabel("Lateral \ndeviation (cm)", fontsize=10)
    axs[1, 0].set_ylabel("Speed \ndeviation (m/s)", fontsize=10)
    axs[0, 0].yaxis.set_tick_params(labelsize=9)
    axs[1, 0].yaxis.set_tick_params(labelsize=9)

    axs[0, 0].yaxis.set_major_locator(plt.MaxNLocator(5))
    axs[1, 0].yaxis.set_major_locator(plt.MaxNLocator(4))
    
    
    fig.legend(labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.95), fontsize=8)
    
    plt.tight_layout()
    name = f"{base_path}Imgs/RacelineDeviation_{set_number}_{map_name.upper()}"
    std_img_saving(name, True)
    
    
make_deviation_bar_plot()