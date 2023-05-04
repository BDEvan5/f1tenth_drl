import numpy as np

def make_comparison_table():
    set_n = 3
    name = "comparison"
    folder = f"NetworkFitting/{name}_{set_n}/"
    name_keys = ["fullPlanning", "trajectoryTrack", "endToEnd"]
    labels = ["Full planning", "Trajectory tracking", "End-to-End"]
    
    train_losses, train_std = [], []
    test_losses, test_std = [], []
    
    with open(folder + f"LossResults/{name}_LossResults.txt", "r") as f:
        lines = f.readlines()
        for l, line in enumerate(lines):
            if l == 0: continue
            line = line.split(",")
            train_losses.append(float(line[1]))
            train_std.append(float(line[2]))
            test_losses.append(float(line[3]))
            test_std.append(float(line[4]))
                
    with open(folder + f"LossResults/{name}_LossResultLatex.txt", "w") as file:
        file.write(f"\t \\toprule \n")
        file.write("\t  \\textbf{Architecture} & \\textbf{Training Loss} & \\textbf{Testing Loss} \\\\ \n")
        file.write(f"\t  \midrule \n")
        
        for i in range(len(labels)):
            file.write(f"\t  {labels[i]} & ".ljust(30))
            file.write(f"{train_losses[i]} $\pm$ {train_std[i]} & ".ljust(25))
            file.write(f"{test_losses[i]} $\pm$ {train_std[i]} ".ljust(25))
            file.write("\\\\ \n")
        else: file.write(f"\t \\bottomrule \n")



make_comparison_table()