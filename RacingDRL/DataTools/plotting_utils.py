import matplotlib.pyplot as plt
import numpy as np
import csv
import glob



def std_img_saving(name, SavePDF=True):

    plt.rcParams['pdf.use14corefonts'] = True

    plt.tight_layout()
    plt.grid(True)
    plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)
    if SavePDF:
        plt.savefig(name + ".pdf", bbox_inches='tight', pad_inches=0)



def load_time_data(folder, map_name=""):
    files = glob.glob(folder + f"Results_*{map_name}*.txt")
    files.sort()
    print(files)
    keys = ["time", "success", "progress"]
    mins, maxes, means = {}, {}, {}
    for key in keys:
        mins[key] = []
        maxes[key] = []
        means[key] = []
    
    for i in range(len(files)):
        with open(files[i], 'r') as file:
            lines = file.readlines()
            for j in range(len(keys)):
                mins[keys[j]].append(float(lines[3].split(",")[1+j]))
                maxes[keys[j]].append(float(lines[4].split(",")[1+j]))
                means[keys[j]].append(float(lines[1].split(",")[1+j]))

    return mins, maxes, means

def load_data_mean_std(folder, map_name=""):
    files = glob.glob(folder + f"Results_*{map_name}*.txt")
    files.sort()
    print(files)
    keys = ["time", "success", "progress"]
    means, stds = {}, {}
    for key in keys:
        means[key] = []
        stds[key] = []
    
    for i in range(len(files)):
        with open(files[i], 'r') as file:
            lines = file.readlines()
            for j in range(len(keys)):
                means[keys[j]].append(float(lines[1].split(",")[1+j]))
                stds[keys[j]].append(float(lines[2].split(",")[1+j]))

    return means, stds

def load_data_mean_std(folder, map_name=""):
    files = glob.glob(folder + f"Results_*{map_name}*.txt")
    files.sort()
    print(files)
    keys = ["time", "success", "progress"]
    means, stds = {}, {}
    for key in keys:
        means[key] = []
        stds[key] = []
    
    for i in range(len(files)):
        with open(files[i], 'r') as file:
            lines = file.readlines()
            for j in range(len(keys)):
                means[keys[j]].append(float(lines[1].split(",")[1+j]))
                stds[keys[j]].append(float(lines[2].split(",")[1+j]))

    return means, stds


def load_repetition_data(folder, map_name=""):
    """loads the data for a repetition set

    Args:
        folder (path): path to a folder
        map_name (str, optional): describes the specific result. Defaults to "".

    Returns:
        _type_: _description_
    """
    files = glob.glob(folder + f"RepetitionSummary_*{map_name}*.txt")
    try:
        file_name = files[0]
    except:
        name_str = folder + f"RepetitionSummary_*{map_name}*.txt"
        print(f"None found: {name_str}")
        return
    print(file_name)
    
    times = []
    successes = []
    progresses = []
    repetition_numbers = [f"{z}" for z in range(10)]
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for l, line in enumerate(lines):
            if l == 0: continue
            if line[0] == "-": continue
            digit = line.split(",")[0][0]
            if digit in repetition_numbers:
                times.append(float(line.split(",")[1]))
                successes.append(float(line.split(",")[2]))
                progresses.append(float(line.split(",")[3]))
                           
    return times, successes, progresses

def load_detail_mean_std(folder, map_name=""):
    file = folder + f"DetailSummaryStatistics{map_name.upper()}.txt"
    keys = ["time", "progress", "distance", "avg_speed", "std_speed", "avg_lateral", "std_lateral", "avg_speedD", "std_speedD", "avg_curvature", "std_curvature"]
    means, stds = {}, {}
    for key in keys:
        means[key] = []
        stds[key] = []
    
    with open(file, 'r') as file:
        lines = file.readlines()
        for j in range(len(keys)):
            means[keys[j]].append(float(lines[1].split(",")[1+j]))
            stds[keys[j]].append(float(lines[2].split(",")[1+j]))

    return means, stds


def load_csv_data(path):
    """loads data from a csv training file

    Args:   
        path (file_path): path to the agent

    Returns:
        rewards: ndarray of rewards
        lengths: ndarray of episode lengths
        progresses: ndarray of track progresses
        laptimes: ndarray of laptimes
    """
    rewards, lengths, progresses, laptimes = [], [], [], []
    with open(f"{path}training_data_episodes.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if float(row[2]) > 0:
                rewards.append(float(row[1]))
                lengths.append(float(row[2]))
                progresses.append(float(row[3]))
                laptimes.append(float(row[4]))

    rewards = np.array(rewards)[:-1]
    lengths = np.array(lengths)[:-1]
    progresses = np.array(progresses)[:-1]
    laptimes = np.array(laptimes)[:-1]
    
    return rewards, lengths, progresses, laptimes

def true_moving_average(data, period):
    if len(data) < period:
        return np.zeros_like(data)
    ret = np.convolve(data, np.ones(period), 'same') / period
    for i in range(period): # start
        t = np.convolve(data, np.ones(i+2), 'valid') / (i+2)
        ret[i] = t[0]
    for i in range(period):
        length = int(round((i + period)/2))
        t = np.convolve(data, np.ones(length), 'valid') / length
        ret[-i-1] = t[-1]
    return ret

def convert_to_min_max_avg(step_list, progress_list, xs):
    """Returns the 3 lines 
        - Minimum line
        - maximum line 
        - average line 
    """ 
    n = len(step_list)

    ys = np.zeros((n, len(xs)))
    for i in range(n):
        ys[i] = np.interp(xs, step_list[i], progress_list[i])

    min_line = np.min(ys, axis=0)
    max_line = np.max(ys, axis=0)
    avg_line = np.mean(ys, axis=0)

    return min_line, max_line, avg_line

def smooth_line(steps, progresses, length_xs=300):
    xs = np.linspace(steps[0], steps[-1], length_xs)
    smooth_line = np.interp(xs, steps, progresses)

    return xs, smooth_line



pp_light = ["#EC7063", "#5499C7", "#58D68D", "#F4D03F", "#AF7AC5", "#F5B041", "#EB984E"]    
pp = ["#CB4335", "#2874A6", "#229954", "#D4AC0D", "#884EA0", "#BA4A00", "#17A589"]
pp_dark = ["#943126", "#1A5276", "#1D8348", "#9A7D0A", "#633974", "#9C640C", "#7E5109"]
pp_darkest = ["#78281F", "#154360", "#186A3B", "#7D6608", "#512E5F", "#7E5109"]

light_blue = "#5DADE2"
dark_blue = "#154360"
light_red = "#EC7063"
dark_red = "#78281F"
light_green = "#58D68D"
dark_green = "#186A3B"

light_purple = "#AF7AC5"
light_yellow = "#F7DC6F"

plot_green = "#2ECC71"
plot_red = "#E74C3C"
plot_blue = "#3498DB"

def plot_error_bars(x_base, mins, maxes, dark_color, w, tails=True):
    for i in range(len(x_base)):
        xs = [x_base[i], x_base[i]]
        ys = [mins[i], maxes[i]]
        plt.plot(xs, ys, color=dark_color[i], linewidth=2)
        if tails:
            xs = [x_base[i]-w, x_base[i]+w]
            y1 = [mins[i], mins[i]]
            y2 = [maxes[i], maxes[i]]
            plt.plot(xs, y1, color=dark_color[i], linewidth=2)
            plt.plot(xs, y2, color=dark_color[i], linewidth=2)

def plot_error_bars_single_colour(x_base, mins, maxes, dark_color, w, tails=True):
    for i in range(len(x_base)):
        xs = [x_base[i], x_base[i]]
        ys = [mins[i], maxes[i]]
        plt.plot(xs, ys, color=dark_color, linewidth=2)
        if tails:
            xs = [x_base[i]-w, x_base[i]+w]
            y1 = [mins[i], mins[i]]
            y2 = [maxes[i], maxes[i]]
            plt.plot(xs, y1, color=dark_color, linewidth=2)
            plt.plot(xs, y2, color=dark_color, linewidth=2)



def plot_color_pallet(pp, name="None"):
    plt.figure(figsize=(10, 1))
    plt.axis('off')
    for i in range(len(pp)):
        plt.plot([i, i], [0, 1], color=pp[i], linewidth=50)
        
    plt.tight_layout
    plt.savefig(f"Data/Imgs/Pallets/color_pallet_{name}.svg", bbox_inches='tight', pad_inches=0)
    # plt.show()
    
science_pallet = ['#0C5DA5', '#FF2C00', '#00B945', '#FF9500', '#845B97', '#474747', '#9e9e9e']
# science_pallet = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
color_blind_pallet = ["#c1272d", "#0000a7", "#eecc16", "#008176", "#b3b3b3"]
    
science_bright = ['EE7733', '0077BB', '33BBEE', 'EE3377', 'CC3311', '009988', 'BBBBBB']
science_bright = [f"#{c}" for c in science_bright]
science_vibrant = ['4477AA', 'EE6677', '228833', 'CCBB44', '66CCEE', 'AA3377', 'BBBBBB']
science_vibrant = [f"#{c}" for c in science_vibrant]

science_high_vis = ["0d49fb", "e6091c", "26eb47", "8936df", "fec32d", "25d7fd"]
science_high_vis = [f"#{c}" for c in science_high_vis]

google = ["#008744", "#0057e7", "#d62d20", "#ffa700"]

color_pallet = science_pallet

if __name__ == '__main__':
    plot_color_pallet(pp, "std")
    plot_color_pallet(pp_dark, "dard")
    plot_color_pallet(pp_light, "pp_light")
    plot_color_pallet(pp_darkest,"darkest")
    
    plot_color_pallet(science_pallet, "science")
    plot_color_pallet(color_blind_pallet, "color_blind")
    
    plot_color_pallet(science_bright, "science_bright")
    plot_color_pallet(science_vibrant, "science_vibrant")
    
    plot_color_pallet(science_high_vis, "science_high_vis")
    plot_color_pallet(google, "google")
    