from matplotlib import pyplot as plt
import numpy as np
import glob
import os

time_idx = 3
progress_idx = 2
success_idx = -1

class VehicleData:
    def __init__(self, vehicle_id, n=3, prefix="Data/Vehicles/Cth_speedMaps/"):
        self.vehicle_id = vehicle_id
        self.prefix = prefix 
        # map_names = ["aut", "esp", "gbr", "mco"]
        map_names = [vehicle_id.split('_')[3]]
        for map_name in map_names:
            self.times = []
            self.success_rates = []
            self.avg_progresses = []
            
            for i in range(n):
                self.process_folder(vehicle_id, i, map_name)
            
            self.save_data(map_name)
        
    def process_folder(self, name, n, map_name):
        folder = self.prefix + name + "_" + str(n) 
        
        try:
            with open(f"{folder}/SummaryStatistics{map_name.upper()}.txt", 'r') as file:
                lines = file.readlines()
                line = lines[4] # first lap is heading
                line = line.split(',')
                
                time = float(line[time_idx])
                if not np.isnan(time): 
                    self.times.append(time)
                    success = float(line[success_idx])
                    self.success_rates.append(success)
                    
                avg_progress = float(line[progress_idx])
                self.avg_progresses.append(avg_progress)
        except:
            print(f"File not opened: {folder}")
            
    def save_data(self, map_name):
        functions = [np.mean, np.std, np.amin, np.amax]
        names = ["Mean", "Std", "Min", "Max"]
        
        times = np.array(self.times)
        success_rates = np.array(self.success_rates)
        progresses = np.array(self.avg_progresses)

        with open(self.prefix + "Results_" + self.vehicle_id + f"_test{map_name[-3:].upper()}.txt", 'w') as file:
            file.write(f"Metric  , Time              , Success Rate     , Avg Progress    \n")
            for i in range(len(names)):
                file.write(f"{names[i]}".ljust(10))
                if len(times) == 1:
                    file.write(f", {times[0]:14.4f}")
                    file.write(f", {success_rates[0]:14.4f}")
                elif len(times) == 0:
                    file.write(f", nan".rjust(14))
                    file.write(f", nan".rjust(14))
                else:
                    file.write(f", {functions[i](times):14.4f}")
                    file.write(f", {functions[i](success_rates):14.4f}")
                    
                file.write(f", {functions[i](progresses):14.4f} \n")
            


def aggregate_runs(path, n=3):
    vehicle_folders = glob.glob(f"{path}*/")
    vehicle_folders.sort()
    
    print(f"{len(vehicle_folders)} folders found")

    id_list = []
    for j, folder in enumerate(vehicle_folders):
        print(f"Vehicle folder being opened: {folder}")
        
        
        vehicle_name = folder.split("/")[-2]
        vehicle_id = vehicle_name[:-2]
        print(vehicle_id)
        
        if not vehicle_id in id_list and vehicle_id != "_Im":
            id_list.append(vehicle_id)
        
    for i in range(len(id_list)):
        v = VehicleData(id_list[i], n=n, prefix=path)
        # v = VehicleData(id_list[i], n=3, prefix=path)
        



# aggregate_runs("Data/TrajectoryNumPoints_4/", 2)
# aggregate_runs("Data/GameMaps_3/", 3)
# aggregate_runs("Data/PurePursuitMaps_1/", 1)

set_n = 1
# aggregate_runs(f"Data/TrajectoryMaps_{set_n}/", 3)
# aggregate_runs(f"Data/EndMaps_{set_n}/", 3)
# aggregate_runs(f"Data/PlanningMaps_{set_n}/", 3)
# aggregate_runs(f"Data/PurePursuitMaps_{set_n}/", 1)

aggregate_runs(f"Data/FinalExperiment_{set_n}/", 3)