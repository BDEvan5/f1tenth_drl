from matplotlib import pyplot as plt
import numpy as np
from FTenthRacingDRL.DataTools.plotting_utils import *
from matplotlib.ticker import MultipleLocator

def make_repetition_bar_plot():
    train_map = "mco"
    base_path = "Data/"
    set_number = 5
    folder_keys = ["PlanningMaps", "TrajectoryMaps", "EndMaps", "PurePursuitMaps"]

    folder_list = [base_path + folder_keys[i] + f"_{set_number}/" for i in range(4)]
    
    folder_labels = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]
    
    keys = ["time", "success", "progress"]
    title_keys = ["Lap time (s)", "Success (%)", "Avg. Progress (%)"]
    
    t_list = []
    s_list = []
    p_list = []
    
    for f, folder in enumerate(folder_list):
        if f < 3:
            ts, ss, ps = load_repetition_data(folder, f"{train_map}_TAL_8_5_testMCO")
        else:
            ts, ss, ps = load_repetition_data(folder, f"{train_map}_test_8_5")
            for i in range(2):
                ts.append(ts[0])
                ss.append(ss[0])
                ps.append(ps[0])
            
        t_list.append(ts)
        s_list.append(ss)
        p_list.append(ps)
        
    data = [t_list, s_list, p_list]
    data = np.array(data)
    xs = np.arange(4)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    for i in range(4):
        plot_data = data[i]
        axs[i].bar(xs, np.mean(plot_data, axis=1), color=color_pallet, alpha=0.25)
        
        for fl in range(4):
            for d in plot_data[fl]:
                axs[i].plot(fl, d, 'o', color=color_pallet[fl], markersize=8, label=folder_labels[fl])
                # axs[i].plot(fl, d, 'x', color=color_pallet[fl], markersize=10)
            
        
        # axs[i]
        
        axs[i].set_xticks([])
        # axs[i].set_xticks(np.arange(4))
        # axs[i].set_xticklabels(folder_labels)
        axs[i].set_title(title_keys[i])
        axs[i].grid(True)
        axs[i].set_axisbelow(True)
        
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles[::3], labels[::3], loc='center', bbox_to_anchor=(0.5, 1), ncol=4)
            
    plt.tight_layout()
    
    std_img_saving_path = base_path + f"Imgs/repetition_bar_plot_{train_map.upper()}"
    std_img_saving(std_img_saving_path, True)

def make_repetition_bar_plot_article():
    train_map = "mco"
    base_path = "Data/"
    set_number = 5
    folder_keys = ["PlanningMaps", "TrajectoryMaps", "EndMaps", "PurePursuitMaps"]

    folder_list = [base_path + folder_keys[i] + f"_{set_number}/" for i in range(4)]
    
    folder_labels = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]
    
    keys = ["time", "success", "progress"]
    title_keys = ["Lap time (s)", "Success (%)", "Avg. Progress (%)"]
    
    t_list = []
    s_list = []
    p_list = []
    
    for f, folder in enumerate(folder_list):
        if f < 3:
            ts, ss, ps = load_repetition_data(folder, f"{train_map}_TAL_8_5_testMCO")
        else:
            ts, ss, ps = load_repetition_data(folder, f"{train_map}_TAL_8_5_testMCO")
            # ts, ss, ps = load_repetition_data(folder, f"{train_map}_TAL_8_5")
            # ts, ss, ps = load_repetition_data(folder, f"{train_map}_test_8_5")
            for i in range(2):
                ts.append(ts[0])
                ss.append(ss[0])
                ps.append(ps[0])
            
        t_list.append(ts)
        s_list.append(ss)
        p_list.append(ps)
        
    data = [t_list, s_list, p_list]
    data = np.array(data)
    xs = np.arange(4)
    n_keys = 2
    fig, axs = plt.subplots(1, n_keys, figsize=(5, 2))
    for i in range(n_keys):
        plot_data = data[i]
        axs[i].bar(xs, np.mean(plot_data, axis=1), color=color_pallet, alpha=0.25)
        
        for fl in range(4):
            for di, d in enumerate(plot_data[fl]):
                axs[i].plot(fl - 0.2 + di*0.2, d, 'o', color=color_pallet[fl], markersize=7, label=folder_labels[fl])
        
        axs[i].set_xticks([])
        axs[i].set_title(title_keys[i], fontsize=10)
        axs[i].grid(True)
        axs[i].set_axisbelow(True)
        
    axs[0].yaxis.set_major_locator(MultipleLocator(10))
    axs[0].set_ylim([25, 60])
        
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles[::3], labels[::3], loc='center', bbox_to_anchor=(0.5, 1), ncol=4, fontsize=8)
            
    plt.tight_layout()
    
    std_img_saving_path = base_path + f"Imgs/repetition_bar_plot_{train_map.upper()}"
    std_img_saving(std_img_saving_path, True)

def make_repetition_bar_plot_single_folder():
    train_map = "mco"
    set_number = 1
    base_path = f"Data/FinalExperiment_{set_number}/"

    folder_labels = ["Full planning", "Trajectory tracking", "End-to-end", "Classic"]
    key_list = ['Game', "TrajectoryFollower", "endToEnd", "pathFollower"]
    
    keys = ["time", "success", "progress"]
    title_keys = ["Lap time (s)", "Success (%)", "Avg. Progress (%)"]
    
    t_list = []
    s_list = []
    p_list = []
    
    for f, folder in enumerate(key_list):
        if f < 3:
            ts, ss, ps = load_repetition_data(base_path, f"{folder}_{train_map}_TAL_8_{set_number}_testMCO")
        else:
            ts, ss, ps = load_repetition_data(base_path, f"{train_map}_TAL_8_{set_number}_testMCO")
            for i in range(4):
                ts.append(ts[0])
                ss.append(ss[0])
                ps.append(ps[0])
            
        t_list.append(ts)
        s_list.append(ss)
        p_list.append(ps)
        
    data = [t_list, s_list, p_list]
    data = np.array(data)
    xs = np.arange(len(key_list))
    n_keys = 2
    spacing = 0.1
    fig, axs = plt.subplots(1, n_keys, figsize=(5, 2))
    for i in range(n_keys):
        plot_data = data[i]
        axs[i].bar(xs, np.mean(plot_data, axis=1), color=color_pallet, alpha=0.25)
        
        for fl in range(len(key_list)):
            for di, d in enumerate(plot_data[fl]):
                axs[i].plot(fl - 0.2 + di*spacing, d, 'o', color=color_pallet[fl], markersize=7, label=folder_labels[fl])
        
        axs[i].set_xticks([])
        axs[i].set_title(title_keys[i], fontsize=10)
        axs[i].grid(True)
        axs[i].set_axisbelow(True)
        
    axs[0].yaxis.set_major_locator(MultipleLocator(10))
    axs[0].set_ylim([25, 60])
        
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles[::5], labels[::5], loc='center', bbox_to_anchor=(0.5, 1), ncol=4, fontsize=8)
            
    plt.tight_layout()
    
    std_img_saving_path = base_path + f"Imgs/repetition_bar_plot_{train_map.upper()}"
    std_img_saving(std_img_saving_path, True)

# make_repetition_bar_plot()
# make_repetition_bar_plot_article()
make_repetition_bar_plot_single_folder()